"""Pinned LM extraction with word ownership and matched context contrasts."""
import json
import math
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import key, write_table
from features import context_windows, geometry, distances, scalar_features

def encode_regions(words, tokenizer, allow_internal_whitespace=False):
    #This is basically to tokenise the whole thing once, and then assign offsets by overlap with known regions (separator spaces belong to no region).

    if not words or any(not w.strip() or (not allow_internal_whitespace and any(c.isspace() for c in w)) for w in words):
        raise ValueError("Expected nonempty reading regions without whitespace")
    text = " ".join(words)
    spans = []
    start = 0
    for word in words:
        spans.append((start, start + len(word)))
        start += len(word) + 1
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=False, verbose=False)
    if allow_internal_whitespace:
        # This is to ignore whitespace when determining whether a BPE crosses actual region text.
        character_owner = [-1] * len(text)
        for index, (a, b) in enumerate(spans):
            character_owner[a:b] = [index] * (b-a)
        owners = []
        for a, b in encoded["offset_mapping"]:
            candidates = {character_owner[j] for j in range(a,b)
                          if character_owner[j] >= 0 and not text[j].isspace()}
            if len(candidates) > 1 or a == b:
                raise ValueError(f"BPE spans multiple reading regions: {(a,b)}")
            if candidates:
                owner = next(iter(candidates))
            else:
                within = [character_owner[j] for j in range(a,b) if character_owner[j] >= 0]
                owner = within[-1] if within else next((i for i,(_,end) in enumerate(spans) if end > a), len(words)-1)
            owners.append(owner)
        if set(owners) != set(range(len(words))):
            raise ValueError("Some reading regions have no BPE tokens")
        return encoded["input_ids"], owners
    owners = []
    region = 0
    for start, end in encoded["offset_mapping"]:
        while region < len(spans) and spans[region][1] <= start:
            region += 1
        if (region == len(spans) or end <= spans[region][0] or start == end
                or (region + 1 < len(spans) and end > spans[region + 1][0])):
            raise ValueError(f"BPE offset {(start, end)} has ambiguous region ownership")
        owners.append(region)
    if set(owners) != set(range(len(words))):
        raise ValueError("Some reading regions have no BPE tokens")
    return encoded["input_ids"], owners


def score_tokens(ids, model, device, stride=256):
    #This is to score each token once using logits right before its position. 1st window uses the available prefix, and the next windows score only their final stride targets. Positions restart at 0 in each window

    window = model.config.max_position_embeddings
    scores = [float("nan")] * len(ids)
    with torch.inference_mode():
        for begin, end, target_start in context_windows(len(ids), window, stride):
            x = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
            logits = model(input_ids=x, use_cache=False).logits[0]
            local_start = target_start - begin
            loss = torch.nn.functional.cross_entropy(
                logits[local_start - 1:-1].float(), x[0, local_start:], reduction="none")
            scores[target_start:end] = (loss / math.log(2)).cpu().tolist()
    return scores


class BackboneAdapter:
    def __init__(self, backbone):
        self.backbone=backbone
        self.ln_f=backbone.final_layer_norm
    def __call__(self, **kwargs):
        return self.backbone(**kwargs)


class ModelAdapter:
    def __init__(self, model):
        self.native=model
        c=model.config
        self.config=SimpleNamespace(n_layer=c.num_hidden_layers,n_embd=c.hidden_size,
                                    max_position_embeddings=1024)
        self.transformer=BackboneAdapter(model.gpt_neox)
    def __call__(self, **kwargs):
        return self.native(**kwargs)


def extract(rows, tokenizer, model, device, eligible, stride=256, whitespace=False, batch_size=16):
    ids, owners = encode_regions([r['word'] for r in rows], tokenizer, whitespace)
    owners = np.asarray(owners); n = len(rows); layers = model.config.n_layer
    selected = [layers//3, 2*layers//3, layers]
    sums = np.zeros((n, layers+1, model.config.n_embd), np.float64)
    counts = np.zeros(n, int); visits = np.zeros(len(ids), int)
    shortened = np.zeros((n, 3, model.config.n_embd), np.float64)
    relations = np.zeros((n, 9), np.float64)
    included = {i for i,r in enumerate(rows) if key(r) in eligible}
    captured = {}
    def hook(module, args): captured['raw'] = args[0]
    handle = model.transformer.ln_f.register_forward_pre_hook(hook)
    starts = np.array([np.flatnonzero(owners == i)[0] for i in range(n)])
    windows = list(context_windows(len(ids), model.config.max_position_embeddings, stride))
    if len(ids) == 1: windows = [(0,1,1)]
    try:
        with torch.inference_mode():
            for begin,end,target in windows:
                first = 0 if begin == 0 and target == 1 else target
                output = model.transformer(input_ids=torch.tensor([ids[begin:end]], device=device),
                    output_hidden_states=True, use_cache=False, return_dict=True)
                states = np.stack([a[0].float().cpu().numpy() for a in (*output.hidden_states[:-1], captured['raw'])])
                own = owners[first:end]; local = first-begin
                np.add.at(sums, own, states[:,local:].transpose(1,0,2))
                np.add.at(counts, own, 1); visits[first:end] += 1
                tasks = []
                for word in sorted(set(own) & included):
                    positions = np.flatnonzero(owners[begin:end] == word)
                    scored = positions[positions >= local]
                    weight = len(scored)
                    current = states[selected][:,scored].mean(1)
                    predecessors = []
                    for lag in (1,2,3):
                        before = word-lag
                        if before < 0: continue
                        pos = np.flatnonzero(owners[begin:end] == before)
                        if not len(pos) or starts[before] < begin:
                            raise ValueError('Incomplete preceding region in assigned window')
                        predecessors.append(states[selected][:,pos].mean(1))
                    if len(predecessors) < 2: raise ValueError('Eligible row lacks two preceding words')
                    references = [predecessors[0], predecessors[1], np.mean(predecessors,axis=0)]
                    relations[word] += np.concatenate([distances(current,p)[:,0] for p in references])*weight
                    word_start = max(begin, int(starts[word])); word_end = begin+int(positions[-1])+1
                    for index,limit in enumerate((0,16,64)):
                        start = max(begin, word_start-limit)
                        if start == begin:
                            shortened[word,index] += states[-1,scored].sum(0)
                        else:
                            tasks.append((word,index,start,word_end,scored+begin-start))

                tasks.sort(key=lambda t:t[3]-t[2])
                for offset in range(0,len(tasks),batch_size):
                    batch=tasks[offset:offset+batch_size]; width=max(t[3]-t[2] for t in batch)
                    tokens=np.full((len(batch),width),tokenizer.eos_token_id,np.int64)
                    positions=np.zeros_like(tokens);mask=np.zeros_like(tokens)
                    for j,(_,_,start,stop,_) in enumerate(batch):
                        length=stop-start;tokens[j,:length]=ids[start:stop]
                        positions[j,:length]=np.arange(start-begin,stop-begin);mask[j,:length]=1
                    model.transformer(input_ids=torch.tensor(tokens,device=device),
                        attention_mask=torch.tensor(mask,device=device),position_ids=torch.tensor(positions,device=device),
                        use_cache=False,return_dict=True)
                    raw=captured['raw'].float().cpu().numpy()
                    for j,(word,index,_,_,scored) in enumerate(batch):
                        shortened[word,index] += raw[j,scored].sum(0)
                del output, states
    finally: handle.remove()
    if not np.all(visits == 1) or not np.array_equal(counts,np.bincount(owners,minlength=n)):
        raise ValueError('BPE ownership/coverage failure')
    indices=sorted(included); pooled=sums[indices]/counts[indices,None,None]
    short=shortened[indices]/counts[indices,None,None]
    families=geometry(pooled)
    contrast=distances(pooled[:,-1,None],short)
    families['contextualization']=contrast[:,0]
    families['context_dependence']=contrast[:,1:].reshape(len(indices),4)
    rel=relations[indices]/counts[indices,None]
    families['previous_relations']=np.column_stack([rel,rel[:,[2,5,8]]-rel[:,[0,3,6]]])
    return dict(keys=np.array([key(rows[i]) for i in indices],int), n_bpe=counts[indices],
        hidden=pooled[:,selected].astype(np.float32), **families), pooled


def load_model(info, device, cache_dir=None, local_files_only=False):
    torch.manual_seed(2026)
    kwargs = dict(revision=info['revision'], cache_dir=cache_dir,
                  local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(info['model_id'], **kwargs)
    attention = {} if info['attention'] == 'default' else {'attn_implementation': info['attention']}
    native = AutoModelForCausalLM.from_pretrained(
        info['model_id'], dtype=torch.float32, **attention, **kwargs).to(device).eval()
    assert native.config._commit_hash == info['revision']
    model = ModelAdapter(native) if info['model_id'].startswith('EleutherAI/') else native
    assert model.config.n_layer == info['blocks'] and model.config.n_embd == info['hidden_size']
    return tokenizer, model


def probabilities(rows, tokenizer, model, device, whitespace=False):
    ids, owners = encode_regions([r['word'] for r in rows], tokenizer, whitespace)
    scores = score_tokens(ids, model, device, stride=256)
    sums = np.zeros(len(rows))
    np.add.at(sums, owners, scores)
    sums[0] = np.nan
    counts = np.bincount(owners, minlength=len(rows))
    return [dict(item=r['item'], zone=r['zone'], word=r['word'],
                 n_bpe=int(counts[i]), surprisal=float(sums[i])) for i, r in enumerate(rows)]


def extract_dataset(context, sample, tokenizer, model, device, batch_size, whitespace=False):
    eligible = {key(r) for r in sample}
    parts, scalar, probability = [], [], []
    groups = [list(group) for _, group in groupby(context, key=lambda r: key(r)[0])]
    for i, rows in enumerate(groups):
        probability.extend(probabilities(rows, tokenizer, model, device, whitespace))
        if eligible.intersection(key(r) for r in rows):
            arrays, pooled = extract(rows, tokenizer, model, device, eligible,
                                     stride=256, whitespace=whitespace, batch_size=batch_size)
            parts.append(arrays)
            scalar.append(scalar_features(pooled))
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(groups):
            print(f'  contexts {i + 1}/{len(groups)}', flush=True)
    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    merged['scalar'] = np.concatenate(scalar)
    assert merged['keys'].tolist() == [list(key(r)) for r in sample]
    return probability, merged
