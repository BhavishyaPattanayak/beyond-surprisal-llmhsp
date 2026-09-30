"""CORE controls and the frozen scalar/rich feature definitions."""
import numpy as np
from sklearn.decomposition import PCA
from data import key

EPS = 1e-12

def distances(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.stack([1-np.clip(np.sum(a*b, axis=-1)/np.maximum(na*nb, EPS), -1, 1),
                     np.linalg.norm(a-b, axis=-1)/np.maximum(nb, EPS)], axis=-1)


def geometry(h):
    #h has shape [words, embedding+blocks, hidden]. No post final LayerNorm.
    h = np.asarray(h, dtype=np.float64)
    blocks = h.shape[1]-1
    early, middle = blocks//3, 2*blocks//3
    displacement = distances(h[:, -1, None], h[:, [0, early, middle]]).reshape(len(h), -1)
    delta = np.diff(h, axis=1)
    norms = np.linalg.norm(h, axis=-1)
    movement = np.linalg.norm(delta, axis=-1)
    valid = (movement[:, :-1] > EPS) & (movement[:, 1:] > EPS)
    curvature = distances(delta[:, 1:], delta[:, :-1])[..., 0]
    curvature = np.where(valid, curvature, 0.)
    curves = np.column_stack([curvature.mean(1), curvature.max(1),
        *[p.mean(1) for p in np.array_split(curvature, 3, axis=1)], (~valid).mean(1)])
    log_norm = np.log(np.maximum(norms, EPS)); change = np.diff(log_norm, axis=1)
    dynamics = np.column_stack([log_norm[:, 0], log_norm[:, -1], log_norm[:, -1]-log_norm[:, 0],
        abs(change).max(1), *[p.mean(1) for p in np.array_split(log_norm[:, 1:], 3, axis=1)]])
    return dict(displacement=displacement, curvature=curves, norm_dynamics=dynamics)


def context_windows(n_tokens, window, stride=256):
    #This is because we want something like (begin, end, first target) using the surprisal schedule we decided before.
    if not 1 <= stride < window:
        raise ValueError("stride must be between 1 and context window minus 1")
    target_start = 1
    while target_start < n_tokens:
        end = min(n_tokens, window if target_start == 1 else target_start + stride)
        yield max(0, end - window), end, target_start
        target_start = end


class HiddenTransform:
    def __init__(self,components=16):self.components=components
    def fit(self,h):
        self.pcas=[]
        for layer in range(h.shape[1]):
            p=PCA(n_components=self.components,svd_solver='randomized',iterated_power=3,random_state=2026)
            p.fit(np.asarray(h[:,layer],dtype=np.float64));self.pcas.append(p)
        return self
    def transform(self,h):
        scores=[];typical=[]
        for layer,p in enumerate(self.pcas):
            centered=np.asarray(h[:,layer],dtype=np.float64)-p.mean_
            z=centered@p.components_.T;scores.append(z)
            squared=np.sum(centered**2,axis=1)
            floor=.1*np.mean(p.explained_variance_)
            radius=np.sqrt(np.mean(z**2/(p.explained_variance_+floor),axis=1))
            residual=np.sqrt(np.maximum(squared-np.sum(z**2,axis=1),0)/centered.shape[1])
            typical.extend([np.sqrt(squared/centered.shape[1]),radius,residual])
        return dict(zip(['layer_early','layer_middle','layer_late'],scores),
                    layer_combination=np.column_stack(scores),typicality=np.column_stack(typical))


def prepare(features, train, validation):
    transform = HiddenTransform(16).fit(features['hidden'][train])
    training = {k: v[train] for k, v in features.items() if k != 'hidden'}
    heldout = {k: v[validation] for k, v in features.items() if k != 'hidden'}
    training.update(transform.transform(features['hidden'][train]))
    heldout.update(transform.transform(features['hidden'][validation]))
    return training, heldout


def scalar_features(pooled):
    h = np.asarray(pooled, dtype=np.float64)
    norms = np.linalg.norm(h, axis=2)
    if np.any(norms <= 1e-12):
        raise ValueError('Cosine distance undefined for near-zero state')
    cosine = 1 - np.clip(np.sum(h[:, :-1] * h[:, 1:], axis=2) /
                         (norms[:, :-1] * norms[:, 1:]), -1, 1)
    return np.column_stack([part.mean(axis=1) for part in np.array_split(cosine, 3, axis=1)])


def controls(context, sample, probabilities, settings):
    from itertools import groupby
    lexical = {key(r): r for r in context}
    probability = {key(r): r for r in probabilities}
    assert set(lexical) == set(probability)
    position = {}
    for _, group in groupby(probabilities, key=lambda r: key(r)[0]):
        rows = list(group)
        counts = np.array([int(r['n_bpe']) for r in rows])
        owners = np.repeat(np.arange(len(rows)), counts)
        local = np.zeros(len(owners))
        for begin, end, target in context_windows(len(owners), 1024, 256):
            first = 0 if begin == 0 and target == 1 else target
            local[first:end] = np.arange(first - begin, end - begin)
        prefix = np.r_[0, np.cumsum(counts)[:-1]]
        for i, row in enumerate(rows):
            position[key(row)] = (float(prefix[i]), float(local[owners == i].mean()))
    matrix = []
    names = settings['predictors']['M1']
    for row in sample:
        item, zone = key(row)
        history = [(item, zone - lag) for lag in range(3)]
        words = [lexical[k] for k in history]
        probs = [probability[k] for k in history]
        assert all(w['word'] == p['word'] for w, p in zip(words, probs))
        frequency = float(words[2]['log_frequency'] or 'nan')
        s2 = float(probs[2]['surprisal'])
        # The inherited availability flag was verified to equal zone > 3.
        available = zone > 3
        assert np.isfinite(s2) == available
        values = dict(current_surprisal=float(probs[0]['surprisal']),
                      previous_surprisal=float(probs[1]['surprisal']),
                      word_length=float(row['word_length']),
                      previous_word_length=float(row['prev_word_length']),
                      log_frequency=float(row['log_frequency']),
                      previous_log_frequency=float(row['prev_log_frequency']),
                      n_bpe=float(probs[0]['n_bpe']), previous_n_bpe=float(probs[1]['n_bpe']),
                      lag2_length=float(words[2]['word_length']),
                      lag2_frequency=frequency if np.isfinite(frequency) else 0.,
                      lag2_n_bpe=float(probs[2]['n_bpe']),
                      lag2_frequency_available=float(np.isfinite(frequency)),
                      lag2_surprisal=s2 if available else 0.,
                      lag2_context_available=float(available))
        for prefix, word in zip(('current', 'previous'), [w['word'] for w in words[:2]]):
            flags = [any(c in word for c in '.!?'), ',' in word,
                     any(c in word for c in ':;'), any(c in word for c in '\"“”'),
                     any(c.isdigit() for c in word), bool(word) and word[0].isupper()]
            for name, value in zip(('terminal_punctuation', 'comma', 'colon_semicolon',
                                    'double_quote', 'digit', 'initial_uppercase'), flags):
                values[f'{prefix}_{name}'] = float(value)
        before, local = position[key(row)]
        values.update(log1p_region_position=np.log1p(zone), log1p_prior_bpe_count=np.log1p(before),
                      model_position_div1024=local / 1024,
                      model_position_div1024_squared=(local / 1024) ** 2)
        matrix.append([values[name] for name in names])
    result = np.array(matrix)
    assert np.isfinite(result).all()
    return result
