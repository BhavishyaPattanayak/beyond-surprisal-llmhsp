# Prepared data

The four gzip-compressed UTF-8 TSVs are the fixed starting point of this submission. They retain original serialized values, not newly filtered or reaggregated outcomes. No historical predictions, residuals, surprisal values or hidden states are included.

| Dataset | Context rows | Evaluated rows | Dependent variable |
|---|---:|---:|---|
| Natural Stories | 10,256 | 10,216 | Participant-mean self-paced RT, ms |
| GECO | 56,410 | 30,155 | Participant-mean valid gaze duration, ms |

Context columns are `item`, `zone`, `word`, `word_length`, `log_frequency`. Contexts include all words in order; `item` is a story for Natural Stories and the mapped `(PART, TRIAL)` context for GECO. GECO `zone` follows the authoritative observed sequence. Empty log frequencies are missing, not zero. Length counts displayed characters. Frequency is the original `ln(1+count)` control.

Sample columns are `item`, `zone`, `word`, `fold`, `observed_rt`, `word_length`, `prev_word_length`, `log_frequency`, `prev_log_frequency`. Folds are fixed IDs 0–9. Current/previous lexical values preserve the serialized final design; lag-two lexical values come from the full context table. Missing lag-two frequency is encoded zero plus an availability flag. Do not remove excluded context words before extracting LM features.

The Natural Stories projection comes from the project's human-predictor and joined analysis tables; the GECO projection comes from its processed-region table. Samples/folds/outcomes match the final CORE cases. The pipeline starts here: it does not reconstruct historical exclusion or exploration stages.

Natural Stories human data use supplied processed RTs aggregated over participants. The original eligible cohort required complete lexical and LM history/features. GECO uses `MonolingualReadingData.xlsx`, sheet `DATA`; positive finite observed durations are averaged, with no zero imputation for skips. Numeric/boolean Excel words retain the original displayed conversion. Its lexical lookup used Natural Stories Google Books `.word` counts with case-sensitive matching and outer punctuation stripping; unmatched frequencies remain missing. Frozen evaluation membership preserves the original exclusions across all four models.

Sources and citations:

- [Natural Stories](https://github.com/languageMIT/naturalstories), commit `4700daad696e942f5aba23c957a7423d0de66612`; Futrell et al. (2021), *The Natural Stories corpus*. Upstream inputs: `naturalstories_RTS/all_stories.tok`, `naturalstories_RTS/processed_RTs.tsv`, `freqs/freqs-1.tsv`. Dataset license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). These prepared tables are derived projections/aggregations.
- [GECO](https://expsy.ugent.be/downloads/geco/); Cop, Dirix, Drieghe & Duyck (2017), *Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading*. The large original workbook is available from the source; it is not needed for execution from the bundled prepared tables. Source corpus/text rights remain with their respective holders.

Model weights are obtained automatically from Hugging Face at the immutable revisions in `config/models.json`. No raw workbook, raw participant records, downloaded checkpoints or model feature cache is included here.
