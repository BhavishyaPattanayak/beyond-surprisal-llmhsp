"""Final CORE fits using frozen samples and previously selected rich families."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from features import prepare

VARIANTS = ('M0', 'M1', 'M2_scalar', 'M2_rich')


def ridge_predictions(xtrain, ytrain, xvalidation, alphas):
    scaler = StandardScaler().fit(xtrain)
    x = scaler.transform(xtrain)
    xm, ym = x.mean(axis=0), ytrain.mean()
    u, s, vt = np.linalg.svd(x - xm, full_matrices=False)
    coef = vt.T @ ((s[:, None] / (s[:, None] ** 2 + np.asarray(alphas)[None, :])) *
                   (u.T @ (ytrain - ym))[:, None])
    intercept = ym - xm @ coef
    return scaler.transform(xvalidation) @ coef + intercept


def loglik(y, prediction, variance):
    assert np.isfinite(variance).all() and (variance > 0).all()
    return -.5 * (np.log(2 * np.pi * variance) + (y - prediction) ** 2 / variance)


def evaluate(base, scalar, features, sample, selections, settings):
    y = np.array([float(r['observed_rt']) for r in sample])
    folds = np.array([int(r['fold']) for r in sample])
    m0 = [settings['predictors']['M1'].index(c) for c in settings['predictors']['M0']]
    alphas = np.asarray(settings['alphas'])
    predictions = {m: np.full(len(y), np.nan) for m in VARIANTS}
    variances = {m: np.full(len(y), np.nan) for m in VARIANTS}
    parameters = []
    for outer in range(settings['outer_folds']):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        # Note that these families came from an earlier selection and no family selection takes place here

        selected = selections[str(outer)]

        def matrix(name, indices, transformed):
            if name == 'M0':
                return base[np.ix_(indices, m0)]
            if name == 'M1':
                return base[indices]
            if name == 'M2_scalar':
                return np.c_[base[indices], scalar[indices]]
            return np.column_stack([base[indices]] + [transformed[k] for k in selected])

        losses = {m: [] for m in VARIANTS}
        split = KFold(settings['inner_folds'], shuffle=True,
                      random_state=settings['seed'] + outer)
        for tr, va in split.split(train):
            it, iv = train[tr], train[va]
            ft, fv = prepare(features, it, iv)
            for m in VARIANTS:
                pred = ridge_predictions(matrix(m, it, ft), y[it], matrix(m, iv, fv), alphas)
                losses[m].append(np.sum((y[iv, None] - pred) ** 2, axis=0))
        ft, fv = prepare(features, train, test)
        for m in VARIANTS:
            totals = np.array(losses[m]).sum(axis=0)
            choice = int(np.argmin(totals))
            alpha = float(alphas[choice])
            pred = ridge_predictions(matrix(m, train, ft), y[train], matrix(m, test, fv), [alpha])
            predictions[m][test] = pred.ravel()
            variances[m][test] = totals[choice] / len(train)
            parameters.append(dict(fold=outer, variant=m, alpha=alpha,
                                   variance=float(variances[m][test][0])))
        print(f'  outer fold {outer + 1}/10', flush=True)
    scores = {m: loglik(y, predictions[m], variances[m]) for m in VARIANTS}
    assert all(np.isfinite(a).all() for a in predictions.values())
    return predictions, variances, scores, parameters


def summarize(dataset, model, sample, predictions, scores):
    y = np.array([float(r['observed_rt']) for r in sample])
    folds = np.array([int(r['fold']) for r in sample])
    contrasts = dict(S=scores['M1'] - scores['M0'],
                     R_scalar=scores['M2_scalar'] - scores['M1'],
                     R_rich=scores['M2_rich'] - scores['M1'],
                     T_scalar=scores['M2_scalar'] - scores['M0'],
                     T_rich=scores['M2_rich'] - scores['M0'])
    for name in ('scalar', 'rich'):
        np.testing.assert_allclose(contrasts['S'] + contrasts['R_' + name],
                                   contrasts['T_' + name], rtol=0, atol=1e-12)
    summary = dict(dataset=dataset, model=model, N=len(y),
                   **{k: float(v.sum()) for k, v in contrasts.items()})
    summary.update({k + '_per_word': summary[k] / len(y) for k in contrasts})
    for name in ('scalar', 'rich'):
        summary[name + '_percent_over_surprisal'] = (
            100 * summary['R_' + name] / summary['S'] if summary['S'] > 0 else '')
    for k in ('S', 'R_scalar', 'R_rich'):
        summary['positive_' + k + '_folds'] = sum(
            contrasts[k][folds == f].sum() > 0 for f in range(10))
    fold_rows = [dict(dataset=dataset, model=model, fold=f, N=int((folds == f).sum()),
                      **{k: float(v[folds == f].sum()) for k, v in contrasts.items()},
                      **{m + '_ll': float(v[folds == f].sum()) for m, v in scores.items()})
                 for f in range(10)]
    model_rows = [dict(dataset=dataset, model=model, variant=m, N=len(y),
                       total_ll=float(scores[m].sum()), mean_ll=float(scores[m].mean()),
                       rmse=float(np.sqrt(np.mean((y - predictions[m]) ** 2))),
                       r2=float(1 - np.sum((y - predictions[m]) ** 2) / np.sum((y - y.mean()) ** 2)))
                  for m in VARIANTS]
    return summary, fold_rows, model_rows
