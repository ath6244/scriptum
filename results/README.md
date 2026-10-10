# Results files

Every number in the report comes from a file in this folder. All files are written by the scripts in
`scripts/` (the step numbers below refer to the reproduction table in the main README).

**Conventions used in all files**

- Accuracy, macro-F1, recall, shares and drops are **fractions (0–1)**, not percentages.
- Families (classes) are `claude, gemini, llama, openai, qwen`. Where a file stores them as integers,
  0–4 follow that order.
- `condition` is the input the model saw: `output_only` (masked response), `input_only` (prompt) or
  `input_output` (prompt `[SEP]` masked response).
- `seed` is 448, 449 or 450 for neural models and RQ3 subsamples. Baselines are deterministic on fixed
  splits: `baselines_runs.csv` lists seeds 0–2 with identical results, and their predictions are stored
  once (seed 0).
- `_mean` / `_std` are over seeds. `_ci_low` / `_ci_high` are 95% grouped-bootstrap bounds
  (2,000 resamples of test prompts).
- `test_row` is the row's position in the test split, in the order `src/data.py` `load_splits` returns it.
- Run logs (`*_log.txt`) are console output; dataset text has been removed from them.

## Dataset (steps 1, 2, 10)

| File | Contents | Columns |
|---|---|---|
| `prepare_arena_log.txt` | Every filtering step with row counts, the model → family table | – |
| `data_stats.csv` | Rows, output length and domain counts per family | `LLM_name, rows, mean_words, median_words, n_coding, n_creative_writing, n_general, n_math` |
| `language_filter.csv` | Non-English outputs removed per family | `sampled` (before), `code_only_kept` (code-only outputs kept), `dropped_non_english`, `after_language_filter`, `final` (after balancing) |
| `model_composition.csv` | Rows per exact model version | `LLM_name, model_version, rows` |
| `top_model_version.csv` | Most common version in each family | `LLM_name, model_version, rows, share` (of the family) |
| `masking_stats.csv` | Outputs changed by `[MODEL]` masking | `outputs, changed, share_changed` per family |
| `masking_name_hits.csv` | Outputs containing each masked name | rows = name, columns = family |
| `split_counts.csv` | Rows per split, by family and by domain | `value` (family or domain), `group`, `train, val, test` |
| `pii_scan.csv` | Rows flagged by each PII detector | `column` (`LLM_Input`/`LLM_output`), `detector`, `rows_flagged, share` |

## RQ1 / RQ2: baselines, neural models, statistics (steps 3–6)

| File | Contents | Columns |
|---|---|---|
| `baselines_runs.csv` | One row per baseline × condition × seed | `baseline, condition, seed, best_C` (chosen on val), `val_acc, val_macro_f1, test_acc, test_macro_f1` |
| `baselines.csv` | Baseline summary | `test_acc_mean/std, test_macro_f1_mean/std, val_macro_f1_mean, chance_acc` |
| `confusion_tfidf_output.csv` | TF-IDF + LR confusion matrix, output only | rows `true_<family>`, columns `pred_<family>` (counts) |
| `neural_runs.csv` | One row per TextCNN/BiLSTM run | `best_epoch, epochs_run, test_acc, test_macro_f1, params, train_time_s` |
| `neural_history.csv` | Training curve, one row per epoch | `epoch, train_loss, val_acc, val_macro_f1` |
| `neural_summary.csv` | Neural summary per model × condition | means/stds over seeds, `best_epoch_mean, params, train_time_s_mean` |
| `neural_preds.csv` | Test predictions of every neural run | `test_row, y_true, y_pred` (integers 0–4) |
| `comparison.csv` | **Main results table**: all methods × conditions | `method, condition, test_acc_mean/std, test_macro_f1_mean/std` |
| `comparison_flags.csv` | Pairs whose difference is smaller than the larger seed std | `a, b, metric, diff, max_std, meaningful` |
| `comparison_ci.csv` | Main table with bootstrap CIs | `method, condition, n_seeds, acc, acc_ci_low/high, macro_f1, macro_f1_ci_low/high` |
| `mcnemar.csv` | Exact McNemar tests on the same test rows (seed 448) | `a, b, acc_a, acc_b, only_a_right, only_b_right, p_value, significant_0.05` |
| `binomial_input_only.csv` | Exact binomial test of input-only accuracy vs 20% | `method, seed, n_correct, n, acc, ci_low, ci_high` (Clopper-Pearson), `p_value, above_chance_0.05` |
| `acc_by_length.csv` | Output-only accuracy per output-length quartile, per run | `model, seed, quartile, words_min, words_max, n, acc, macro_f1` |
| `length_quartiles.csv` | Family mix and accuracy per length quartile | `share_<family>`; per model `_acc`, `_balanced_acc` (mean per-family recall), `_acc_expected_from_mix` |
| `near_duplicates.csv` | Near-duplicate prompts across splits | `exact_prompt_ids, normalized_groups, groups_spanning_splits, test_rows, test_rows_affected, share_test_rows_affected` |
| `near_duplicate_groups.csv` | Each normalized-prompt group spanning splits | `norm_id, splits, n_rows` (no prompt text) |
| `run_neural_log.txt` | Console output of `run_neural.py` (vocab size, truncation, per-epoch scores) | – |

### `preds/`: per-row test predictions

Row index and labels only, no text. `y_true` and `y_pred` are family names.

| File | Rows for | Key columns |
|---|---|---|
| `baseline_preds.csv` | Majority, length-only LR, TF-IDF + LR × 3 conditions | `method, condition, seed, test_row, y_true, y_pred` |
| `neural_preds.csv` | TextCNN, BiLSTM × 3 conditions × 3 seeds | same as above |
| `rq4_preds_tfidf_lr.csv`, `rq4_preds_textcnn.csv`, `rq4_preds_length_lr.csv` | RQ4 ablation variants | `model, variant, seed, test_row, y_true, y_pred` |

## RQ3: cross-domain (step 7)

| File | Contents | Columns |
|---|---|---|
| `rq3.csv` | One row per model × held-out domain × condition × seed | `condition` (`in_domain`/`lodo`), `acc, macro_f1, n_test, n_train, train_time_s` |
| `rq3_summary.csv` | Mean/std per model × domain; drop = in-domain − LODO, paired by seed | `acc_*`, `macro_f1_*`, `drop_acc_mean/std`, `drop_*_meaningful` (\|drop\| ≥ std), `n_seeds, n_test` |
| `rq3_per_family_runs.csv` | Per-family recall per run | `family, n_test, recall` |
| `rq3_per_family.csv` | Per-family recall summary | `recall_in_domain_mean/std, recall_lodo_mean/std, n_test, low_n` ("low n" if < 30), `drop` |
| `rq3_log.txt` | Console output (training-set size control) | – |

## RQ4: what distinguishes the families (steps 8, 9)

| File | Contents | Columns |
|---|---|---|
| `rq4_runs.csv` | One row per ablation run | `model, variant, seed, acc, macro_f1, train_time_s` |
| `rq4_ablations.csv` | Ablation summary vs the masked reference | `acc, acc_seed_std, acc_ci_*, macro_f1, macro_f1_ci_*, drop_acc, drop_acc_ci_*, drop_macro_f1, drop_macro_f1_ci_*, drop_ci_excludes_0` |
| `rq4_features.csv` | Style features per family | `median_words, ttr_200` (type-token ratio of the first 200 words), `ttr_200_n, mean_sentence_len, headers/bullets/bold/code_blocks/emoji_per_1k` (words), `share_with_emoji, refusal_rate, refusal_pct` |
| `rq4_top_words.csv` | 15 most distinctive word/bigram features per family | `LLM_name, rank, feature, weight` (TF-IDF + LR coefficient) |
| `rq4_tfidf_log.txt`, `rq4_textcnn_log.txt` | Console output of the ablation runs | – |

## Food for thought (step 11)

| File | Contents | Columns |
|---|---|---|
| `food_for_thought.csv` | Main model's prediction for replies outside the dataset | `file, true_source, predicted, correct, true_rank, p_<family>` |
| `food_for_thought_style.csv` | Each reply's style next to the family profiles | `who` (family or reply file), `words, ttr_200, mean_sentence_len, headers/bullets/bold/emoji_per_1k` |
| `food_for_thought_features.csv` | Features pushing toward the predicted vs the true family | `file, pushes_toward, rank, feature, contribution` |
| `food_for_thought_checks.csv` | Training-data checks behind Section 6 | `file, check, value` |
