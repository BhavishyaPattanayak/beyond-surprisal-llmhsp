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

## Results

Ordinary surprisal improved held out reading time prediction over the control model across all 8 dataset model combinations.
Scalar representation features, based on layerwise cosine distances, provided relatively small additional improvements. However, richer hidden representation features produced substantially larger gains.

The table below shows the additional predictive contribution of richer representations relative to that of surprisal, measured using held out log likelihood.

| Model | Natural Stories | GECO |
|---|---:|---:|
| GPT-2 | 52.6% | 56.5% |
| DistilGPT-2 | 50.0% | 77.5% |
| Pythia 160M | 50.6% | 91.4% |
| Pythia 410M | 73.7% | 64.0% |

These percentages are the additional held out log likelihood improvement from richer representations relative to the improvement from surprisal over the control model. They are the percentage of reading time variance explained.



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
