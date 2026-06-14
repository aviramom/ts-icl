# TimeSeriesExam ICL Experiment — End-to-End Guide

TimeSeriesExam (TSE) is a benchmark of question templates about time series properties.
We re-cast each template as a k-shot ICL classification problem:

- **Each template** (`tid`) = one classification task with 2–4 classes (the options)
- **Support set** = k generated TS variants per option, labeled with that option's letter (A, B, C, D)
- **Query** = a new generated TS whose correct label is the template's ground-truth option
- **Goal** = model predicts which option letter the query satisfies

---

## Prerequisites

- `conda activate multits` (or `multits_large` for 27B models)
- `third_party/TimeSeriesExam/` must be present (cloned from the `exam_generation` branch)
- If not cloned yet:

```bash
git clone --branch exam_generation \
  https://github.com/moment-timeseries-foundation-model/TimeSeriesExam.git \
  third_party/TimeSeriesExam
```

---

## Step 1 — Generate Augmented Data (one-time)

The original `qa_dataset.json` has only 1 TS per question.
This script generates `num_variants` TS per option per template using the TSE generation code.

```bash
python scripts/generate_tse_augmented.py \
  --tse_repo third_party/TimeSeriesExam \
  --output qa_dataset_augmented.json \
  --num_variants 10 \
  --ts_length 256 \
  --seed 42
```

| Argument | Default | Meaning |
|---|---|---|
| `--num_variants` | 10 | TS variants per option per template (~7 train, ~3 test) |
| `--ts_length` | 256 | Length of each generated time series |
| `--seed` | 42 | Reproducibility seed |
| `--tids` | all | Comma-separated list to generate only specific templates, e.g. `--tids 1,3,65` |

**Output format** — each entry in `qa_dataset_augmented.json`:
```json
{
  "tid": 1,
  "question": "What is the type of the trend...",
  "option_names": ["Exponential", "Linear", "No Trend"],
  "category": "Pattern Recognition",
  "subcategory": "Trend Recognition",
  "difficulty": "easy",
  "is_two_series": false,
  "ts_variants": {
    "Linear":      [[256 floats], [256 floats], ...],
    "Exponential": [[256 floats], ...],
    "No Trend":    [[256 floats], ...]
  }
}
```

For **two-series templates** (`is_two_series: true`), each variant is `[[ts1_floats], [ts2_floats]]`.

---

## Step 2 — Verify the Generated File

```bash
python -c "
import json
with open('qa_dataset_augmented.json') as f:
    data = json.load(f)
for e in data[:3]:
    counts = {k: len(v) for k, v in e['ts_variants'].items()}
    print(f\"tid={e['tid']}  options={e['option_names']}  variants={counts}\")
print(f'Total templates: {len(data)}')
"
```

---

## Step 3 — Smoke Test (no GPU, no SLURM)

```bash
# Quick 3-variant generation for testing
python scripts/generate_tse_augmented.py \
  --num_variants 3 --output qa_dataset_augmented.json --tids 1,3,65

bash smoke_tests/test_tse_smoke.sh
```

Or run directly:

```bash
python run_icl.py \
  --task_id icl_tse_1 \
  --method random_baseline \
  --num_shots 1 \
  --picking_strategy random \
  --use_label_desc 1 \
  --tse_data_path qa_dataset_augmented.json \
  --random_seed 0 \
  --display_samples 3
```

Expected output: prompt shows the question + options A/B/C, predictions show `A`/`B`/`C` labels,
metrics are printed. Random baseline on 3 classes should be ~0.33 balanced accuracy.

---

## Step 4 — Single Template with a Real Model (GPU)

```bash
python run_icl.py \
  --task_id icl_tse_1 \
  --method Qwen/Qwen3-4B-Instruct-2507 \
  --num_shots 1 \
  --picking_strategy random \
  --use_label_desc 1 \
  --tse_data_path qa_dataset_augmented.json \
  --random_seed 0 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --display_samples 3
```

`--task_id` is always `icl_tse_{tid}` where `tid` is the integer template id.

---

## Step 5 — Full Benchmark (SLURM)

Generate the full augmented file first (`--num_variants 10`), then:

```bash
bash experiment_scripts/icl_tse_full.sh
```

Environment variables to override defaults:

```bash
METHOD="bytedance-research/ChatTS-8B" \
NUM_SHOTS=1 \
EXP_ID="tse_icl_chatts" \
  bash experiment_scripts/icl_tse_full.sh
```

This submits one SLURM job per `(tid, seed)` pair (all templates × 8 seeds).
Results are saved under `outputs/` and logged to W&B if `--use_wandb 1`.

---

## Step 6 — Aggregate Results

```bash
python evaluations/tse_aggregate_results.py \
  --results_dir outputs/ \
  --augmented_path qa_dataset_augmented.json \
  --method Qwen/Qwen3-4B-Instruct-2507
```

Prints balanced accuracy broken down by **difficulty** (easy/medium/hard)
and **category** (Anomaly Detection, Pattern Recognition, etc.).

---

## Pipeline Code Map

```
Step 1 — Data generation
  scripts/generate_tse_augmented.py
    Calls TSE generation API per option (using third_party/TimeSeriesExam/).
    Generates num_variants TS per option, groups by option index, saves JSON.

Step 2 — Dataset loading (per template)
  data_provider/dataset_tse.py  →  TimeSeriesExamDataset
    Loads one tid from qa_dataset_augmented.json.
    Splits variants into train (support) / test (query) by test_fraction.
    Assigns letters A, B, C, D to options in template order.
    Exposes: label_names, is_two_series, desc (question text + option legend).

Step 3 — Support set selection
  picking_strategy.py  →  get_support_set()
    Picks k examples per class from the train split.
    Use 'random' or 'first' for TSE. Medoid not supported for two-series.

Step 4 — Prompt building
  data_provider/icl_dataset.py  →  ICLUCRDataset.from_ucr_dataset()
    Maps integer labels → A/B/C letters (via dataset.label_names).
    Calls _build_input() for single-TS or _build_input_two_series() for pairs.
    Substitutes <ts><ts/> placeholders with numeric arrays (combined mode).
  utils/formatting.py  →  icl_classification_format()
    Assembles final prompt:
      "Time Series Classification.\n{question+options}\n{examples}\n{query}\nReturn ONLY..."

Step 5 — Model inference
  models/instruct_model.py    →  InstructModel (Qwen3-4B, etc.)
  models/chatts_model.py      →  ChatTSHFWrapper
  models/baselines.py         →  RandomBaseline, KNNBaseline, etc.
  (any model registered in utils/model.py works — TSE needs no model changes)

Step 6 — Label extraction & metrics
  evaluations/icl_ucr_eval.py  →  run_evaluation_icl_ucr()
    _extract_predicted_label(): matches response → A/B/C via exact match + regex
    Computes balanced_accuracy, F1, precision, recall.
    Saves results to outputs/icl_tse_{tid}_{n}_{model}_exp_{id}.json

Step 7 — Cross-template aggregation
  evaluations/tse_aggregate_results.py
    Reads per-tid JSONs from outputs/, groups by difficulty and category.
    Reports mean balanced accuracy per group.

Entry point
  run_icl.py
    Routes icl_tse_* → TimeSeriesExamDataset (lines ~47-62).
    Builds DataLoader, loads model, runs eval, saves JSON and W&B logs.
```

---

## Key Design Details

| Topic | Detail |
|---|---|
| **task_id format** | `icl_tse_{tid}` where `tid` is the integer template id |
| **Class labels in prompt** | Letters A, B, C, D assigned in option order from the template |
| **Description injection** | `--use_label_desc 1` injects question text + option legend into prompt |
| **Two-series questions** | Prompt shows `Time Series 1` / `Time Series 2` per example; use `random` or `first` picking only (medoid fails for tuple TS) |
| **Train/test split** | `--tse_test_fraction 0.3` (default); split is seeded by `--random_seed` |
| **Options string format** | Formatted as `[A, B, C]` (no quotes) so `_extract_predicted_label` can match letter labels |
| **UCR backward compat** | All UCR experiments (`icl_ucr_*`) are unchanged; TSE routing is additive |

---

## TSE Template Categories

| Category | Subcategories | is_two_series |
|---|---|---|
| Pattern Recognition | Trend, Cycle, Stationarity, Regime Switching, AR/MA, First Two Moments | No |
| Anolmaly Detection | General Anomaly Detection | No |
| Noise Understanding | Signal to Noise Ratio | No |
| Similarity Analysis | Shape (flip, lag, scale) | Yes |
| Causality Analysis | Granger Causality | Yes |
