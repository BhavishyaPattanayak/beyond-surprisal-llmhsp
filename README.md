# Hidden representations beyond surprisal

Code for the project *Beyond Surprisal: Predicting Human Reading Times from Transformer Hidden Representations*.

The analysis tests whether hidden representations from GPT-2, DistilGPT-2, Pythia 160M and Pythia 410M improve prediction of human reading times beyond ordinary surprisal and lexical and positional controls.

## Data

The analysis uses Natural Stories and GECO.

Prepared versions of the data used for the final analysis are included in `data/`. Natural Stories contains 10,216 evaluated word regions and GECO contains 30,155 evaluated word regions. The complete context sequences are retained so that language model features can be extracted using the original context.

The outer folds and rich representation families used in the final analysis are frozen in `config/`.

## Models

The final comparison contains four models:

- GPT-2
- DistilGPT-2
- Pythia 160M
- Pythia 410M

Exact model revisions and extraction settings are stored in `config/models.json`.

The evaluated models are:

- M0: lexical, tokenization, punctuation and position controls
- M1: M0 + ordinary surprisal
- M2 scalar: M1 + three layerwise cosine distance summaries
- M2 rich: M1 + richer hidden representation features

Rich representation families were selected separately within each outer training set during an earlier entropy conditioned analysis. Their selections are frozen here. Entropy is not included in the final models.

## Running the analysis

Python 3.12 is recommended.

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Run extraction and evaluation:

```sh
python run.py --stage all --device mps
```

For CPU:

```sh
python run.py --stage all --device cpu
```

If features have already been extracted:

```sh
python run.py --stage evaluate
```

Model checkpoints are downloaded automatically. Generated features are stored under `cache/`.

Final results are written to:

```text
outputs/
├── comparison.tsv
├── fold_comparison.tsv
└── model_scores.tsv
```

Reference results from the analysis used for the poster and report are included under `reference/`.

## Files

- `run.py` — main entry point
- `data.py` — data loading
- `extraction.py` — model extraction and surprisal
- `features.py` — representation features and controls
- `evaluation.py` — regression and held out likelihood evaluation
- `config/` — model settings and frozen analysis specifications
- `data/` — prepared Natural Stories and GECO inputs
- `reference/` — final reference results

Small floating point differences can occur across CUDA, MPS and CPU backends.