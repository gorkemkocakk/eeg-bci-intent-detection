# Stieger2021 EEG Control vs Non-Control Pipeline

## Project Purpose

This repository contains a single-subject, multi-session EEG pipeline for the
Stieger2021 pseudo-online control vs non-control classification task.

- Task: pseudo-online `control` vs `non-control`
- Label `0`: ITI / non-control
- Label `1`: feedback / control
- Cue periods are excluded from classification
- Main evaluation: LOSO-session, leave one session out
- Primary metric: ROC-AUC, with balanced accuracy reported alongside it

## Critical Output And Data Warning

Do not commit generated project outputs.

- Do not commit `outputs/`.
- Do not commit generated CSV, NPZ, logs, figures, or model artifacts.
- Do not use `git add .` in this repo.
- `outputs/` is ignored intentionally by `.gitignore`.
- Local `outputs/` folders may be stale, incomplete, or from old/buggy runs.
- Do not infer clean pipeline status from local `outputs/label_tables/`,
  `outputs/window_data/`, `outputs/window_data_wideband/`, or old
  `outputs/ablation_results/`.
- In the current local repo, only `outputs/csp_ablation_results_original/`
  should be treated as trusted manually downloaded Kaggle CSP ablation CSVs.

For clean reruns, regenerate outputs from the fixed code and verify them with:

```powershell
python check_project_outputs.py
```

This validation should pass only after clean label, normal-window, and wideband
window outputs have been generated for all 11 sessions. It may fail on local
machines if only partial or old outputs are present.

## Basic Pipeline Commands

Per-session manual example:

```powershell
python parse_trials.py 1
python build_labels.py 1
python windowing.py 1
python features_bandpower.py 1
```

All-session normal pipeline:

```powershell
python run_batch_pipeline.py
```

Wideband windows for 5-band CSP:

```powershell
python windowing_wideband.py 1
python windowing_wideband.py 2
python windowing_wideband.py 3
python windowing_wideband.py 4
python windowing_wideband.py 5
python windowing_wideband.py 6
python windowing_wideband.py 7
python windowing_wideband.py 8
python windowing_wideband.py 9
python windowing_wideband.py 10
python windowing_wideband.py 11
```

Output validation, to run in the actual clean output environment:

```powershell
python check_project_outputs.py
```

5-band CSP baseline:

```powershell
python run_cross_session_csp_5band.py
```

CSP component ablation:

```powershell
python run_ablation_csp_components.py
```

Targeted CSP component ablation:

```powershell
python run_ablation_csp_components.py --components 8
python run_ablation_csp_components.py --components 4 8
python run_ablation_csp_components.py 1 3 --components 8
```

Summarize clean ablation outputs generated in the current run:

```powershell
python summarize_csp_ablation.py --results-dir outputs/ablation_results
```

Summarize downloaded trusted Kaggle ablation CSVs:

```powershell
python summarize_csp_ablation.py --results-dir outputs/csp_ablation_results_original
```

## Expected Output Folders

Generated outputs are written under `outputs/`, which is intentionally ignored.
Common output folders are:

- `outputs/label_tables/`
- `outputs/window_data/`
- `outputs/window_data_wideband/`
- `outputs/ablation_results/`
- `outputs/csp_ablation_results_original/` for manually downloaded trusted
  Kaggle CSP ablation CSVs
- `outputs/xai_results_recovered/` for recovered report-level XAI summary CSVs

## Current Confirmed CSP Component Ablation Result

The values below come from the trusted manually downloaded Kaggle ablation CSVs
in `outputs/csp_ablation_results_original/`. They should not be inferred from
arbitrary local old outputs.

| CSP components | Mean ROC-AUC | Mean Balanced Accuracy | Fold count |
| --- | ---: | ---: | ---: |
| 2 | 0.904399 | 0.718756 | 11 |
| 4 | 0.935139 | 0.791548 | 11 |
| 6 | 0.946548 | 0.802168 | 11 |
| 8 | 0.949027 | 0.810928 | 11 |

Eight CSP components had the best observed exploratory ablation result. Four
components remains the original/default strong baseline. The 8-component setup
should be described as an exploratory improved candidate, not as a
pre-registered final model.

`analyze_component_stats.py` summarizes paired 4-vs-8 fold deltas with a
Wilcoxon signed-rank test, bootstrap CI, and effect size. This is supporting
exploratory analysis only; nested LOSO would be needed for unbiased component
selection.

## XAI / Band Importance Diagnostic

`analyze_csp_band_importance.py` runs LOSO-session diagnostics for the 5-band
CSP + LDA model. For each fold it fits CSP, the scaler, and LDA on train
sessions only, evaluates the held-out session, and can compare fold metrics
against existing CSP ablation CSVs. It also supports band-level permutation
importance.

Use band permutation importance as the primary XAI diagnostic. LDA coefficient
magnitude is retained only as a secondary post-hoc diagnostic because it can be
numerically unstable and should not be treated as the main scientific evidence.
In recovered results, gamma can appear very large by coefficient magnitude, but
it has low permutation importance.

Small smoke run:

```powershell
python analyze_csp_band_importance.py --input-dir outputs/window_data_wideband --output-dir outputs/xai_results --components 4 --test-sessions 1 --include-permutation --permutation-repeats 5 --compare-ablation-dir outputs/ablation_results
```

Full diagnostic run:

```powershell
python analyze_csp_band_importance.py --input-dir outputs/window_data_wideband --output-dir outputs/xai_results --components 4 8 --include-permutation --permutation-repeats 5 --compare-ablation-dir outputs/ablation_results
```

Recovered artifact caveat: `outputs/xai_results_recovered/` contains
report-level recovered summary CSVs. The full raw XAI output folders for c4/c8
were lost after a Kaggle runtime restart. These recovered summaries are suitable
for cautious report-level discussion, but they are not a complete raw artifact
archive. For full reproducibility, repeat the XAI run and save artifacts
immediately.

Recommended reporting language: the 4-component model shows alpha and beta
contribution, while the 8-component model shows stronger beta-band dominance.
This is a post-hoc diagnostic, not causal proof. The 8-component setup remains
the best observed exploratory component setting, not a final unbiased model
choice.

## Utility Scripts

Selected experiment and analysis scripts write `manifest.json` into their output
directory. The manifest records the command, git commit/branch, runtime
environment, and run settings for reproducibility.

`check_project_outputs.py` validates that clean project outputs are present and
that normal and wideband label counts match session by session. Run it in the
environment where all 11 clean label/window/wideband outputs have been
generated.

`summarize_csp_ablation.py` summarizes existing CSP component ablation CSVs. It
does not train models. For local trusted downloaded ablation results, use:

```powershell
python summarize_csp_ablation.py --results-dir outputs/csp_ablation_results_original
```

## Methodological Guardrails

- Do not fit supervised transforms before splitting.
- Do not shuffle sessions across the LOSO split.
- Fit scalers only on the training side of each fold.
- Fit CSP only on the training side of each fold.
- Fit feature selection only on the training side of each fold.

## Main Files

- `parse_trials.py`: parse Stieger2021 trial timing into trial tables
- `build_labels.py`: build ITI vs feedback labels
- `windowing.py`: create normal-band windows
- `windowing_wideband.py`: create wideband windows for 5-band CSP
- `features_bandpower.py`: create bandpower features
- `run_batch_pipeline.py`: run the normal per-session output pipeline
- `run_cross_session_csp_5band.py`: run the 5-band CSP LOSO baseline
- `run_ablation_csp_components.py`: run CSP component ablations
- `run_ablation_fs_k.py`: run feature-selection-k ablations
- `check_project_outputs.py`: validate clean generated outputs
- `summarize_csp_ablation.py`: summarize existing ablation CSVs
- `analyze_csp_band_importance.py`: run post-hoc CSP band diagnostics
- `analyze_component_stats.py`: summarize paired CSP component ablation deltas from existing CSVs
