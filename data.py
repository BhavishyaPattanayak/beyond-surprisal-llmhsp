"""Prepared inputs and frozen analysis specifications."""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def key(row):
    return int(row['item']), int(row['zone'])


def read_table(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def configuration():
    def read(name):
        return json.loads((ROOT / 'config' / name).read_text())
    return (read('analysis.json'), read('models.json'),
            read('frozen_rich_family_selections.json')['cases'])


def load_data(dataset, settings):
    context = read_table(ROOT / 'data' / f'{dataset}_context.tsv.gz')
    sample = read_table(ROOT / 'data' / f'{dataset}_sample.tsv.gz')
    assert len(context) == settings['context_counts'][dataset]
    assert len(sample) == settings['sample_counts'][dataset]
    assert [key(r) for r in context] == sorted(set(key(r) for r in context))
    assert [key(r) for r in sample] == sorted(set(key(r) for r in sample))
    words = {key(r): r['word'] for r in context}
    assert all(words[key(r)] == r['word'] for r in sample)
    assert set(int(r['fold']) for r in sample) == set(range(10))
    return context, sample
