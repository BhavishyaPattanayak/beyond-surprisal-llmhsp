"""Extract features and evaluate the frozen final CORE analysis."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True

import numpy as np
from data import ROOT, configuration, key, load_data, read_table, write_table
from features import controls
from evaluation import evaluate, summarize


def fingerprint(dataset):
    files = sorted((ROOT / 'config').glob('*.json'))
    files += sorted((ROOT / 'data').glob(f'{dataset}_*.tsv.gz'))
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for name in ('extraction.py', 'features.py'):
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'extract', 'evaluate'], default='all')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--model', choices=['gpt2', 'distilgpt2', 'pythia-160m', 'pythia-410m'])
    parser.add_argument('--dataset', choices=['natural_stories', 'geco'])
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    settings, models, selections = configuration()
    datasets = [args.dataset] if args.dataset else list(settings['sample_counts'])
    names = [args.model] if args.model else list(models)
    if args.stage in ('all', 'extract'):
        import torch
        from extraction import load_model, extract_dataset
        if args.device == 'mps' and not torch.backends.mps.is_available():
            parser.error('MPS is unavailable; use --device cpu.')
        for name in names:
            tokenizer = model = None
            for dataset in datasets:
                directory = args.cache_dir / f'{dataset}_{name}'
                signature = fingerprint(dataset)
                if (directory / 'complete.json').exists():
                    meta = json.loads((directory / 'complete.json').read_text())
                    if meta['fingerprint'] != signature:
                        raise ValueError(f'Stale cache: choose a new --cache-dir for {directory}')
                    continue
                if tokenizer is None:
                    tokenizer, model = load_model(models[name], args.device)
                context, sample = load_data(dataset, settings)
                print(f'Extracting {dataset}/{name}', flush=True)
                probability, arrays = extract_dataset(
                    context, sample, tokenizer, model, args.device,
                    models[name]['batch_size'], whitespace=dataset == 'geco')
                directory.mkdir(parents=True, exist_ok=True)
                write_table(directory / 'probabilities.tsv', probability)
                np.savez(directory / 'features.npz', **arrays)
                (directory / 'complete.json').write_text(json.dumps(dict(
                    fingerprint=signature, model=models[name], device=args.device), indent=2) + '\n')
            del model, tokenizer
            if args.device == 'mps':
                torch.mps.empty_cache()
    if args.stage in ('all', 'evaluate'):
        summaries, folds, metrics = [], [], []
        for dataset in datasets:
            context, sample = load_data(dataset, settings)
            for name in names:
                directory = args.cache_dir / f'{dataset}_{name}'
                if not (directory / 'complete.json').exists():
                    raise FileNotFoundError(f'Run --stage extract first: {directory}')
                meta = json.loads((directory / 'complete.json').read_text())
                if meta['fingerprint'] != fingerprint(dataset):
                    raise ValueError(f'Cache does not match this analysis: {directory}')
                probability = read_table(directory / 'probabilities.tsv')
                with np.load(directory / 'features.npz', allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                np.testing.assert_array_equal(arrays.pop('keys'), [key(r) for r in sample])
                base = controls(context, sample, probability, settings)
                np.testing.assert_array_equal(arrays.pop('n_bpe'),
                                              base[:, settings['predictors']['M1'].index('n_bpe')])
                scalar = arrays.pop('scalar')
                assert scalar.shape == (len(sample), 3)
                assert all(np.isfinite(a).all() for a in arrays.values())
                print(f'Evaluating {dataset}/{name}', flush=True)
                pred, variance, scores, _ = evaluate(
                    base, scalar, arrays, sample, selections[f'{dataset}_{name}'], settings)
                summary, fold_rows, model_rows = summarize(dataset, name, sample, pred, scores)
                summaries.append(summary)
                folds.extend(fold_rows)
                metrics.extend(model_rows)
        write_table(args.output_dir / 'comparison.tsv', summaries)
        write_table(args.output_dir / 'fold_comparison.tsv', folds)
        write_table(args.output_dir / 'model_scores.tsv', metrics)
        print(f'Results: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
