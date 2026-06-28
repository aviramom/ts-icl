# RQ1: Does ICL Help? Official TSE Format — Zero-Shot vs k=1 ICL

**Experiment:** `tse_rq1_official_icl` (TSE, ~98 templates)  
**Setup:** k=0 (zero-shot) vs k=1 (ICL) · random picking · 3 seeds (0, 3, 6) · 3 models  
**Arg:** `--prompt_format tse_official --num_shots {0, 1}`

---

## Motivation

We want to show that in-context labeled examples (ICL) improve performance on TSE tasks
compared to zero-shot — and that this improvement is consistent across model architectures
(text-only, TS patch-embedding, VL). This addresses **RQ1**: *does ICL help TSLM tasks?*

Unlike prior ablations (e.g., `tse_prompt_format_ablation_qwenvl`) that used the project's
own "Time Series Classification" framing, this experiment uses the **official TimeSeriesExam
MCQ format** as the zero-shot baseline, making the comparison directly comparable to published
TSE benchmark results.

---

## The 2 Conditions

| # | `--num_shots` | Has examples? | Format hint |
|---|---|---|---|
| 1 | 0 (zero-shot) | No | Official format_hint |
| 2 | 1 (ICL k=1) | Yes (1 per class) | Official format_hint |

Both conditions use `--prompt_format tse_official`.

---

## Prompt Samples

Both examples use **tid=1** (trend type: Linear / Exponential / No Trend), **seed=0**.
TS values are truncated; full series are 256 values.

---

### Condition 1 — Zero-Shot (`--num_shots 0`)

```
What is the type of the trend of the given time series?

A) Linear
B) Exponential
C) No Trend

Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]

Please answer the question and provide the correct option letter, e.g., A), B), C), D), and option content at the end of your answer. All information need to answer the question is given. If you are unsure, please provide your best guess.
```

---

### Condition 2 — ICL k=1 (`--num_shots 1`)

```
Here are some labeled examples:

Example 1 Time Series: [0.2379, -0.3516, -1.0774, -1.0349, -1.1402, 1.4715, ... (256 values total)]
Answer: A) Linear

Example 2 Time Series: [-0.1419, 0.5779, 1.1156, 0.9655, 1.0936, 1.8459, ... (256 values total)]
Answer: B) Exponential

Example 3 Time Series: [5.2145, 5.1691, 5.1507, 5.0919, 5.1921, 5.1585, ... (256 values total)]
Answer: C) No Trend

Now answer the following:

What is the type of the trend of the given time series?

A) Linear
B) Exponential
C) No Trend

Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]

Please answer the question and provide the correct option letter, e.g., A), B), C), D), and option content at the end of your answer. All information need to answer the question is given. If you are unsure, please provide your best guess.
```

---

## Models

| Model | Modality | Runner |
|-------|----------|--------|
| `Qwen/Qwen3-8B-Instruct` | Text-only (TS as numeric array) | `run_single_task_gpu.sh` |
| `bytedance-research/ChatTS-8B` | TS patch embeddings + text | `run_single_task_gpu.sh` |
| `Qwen/Qwen3-VL-8B-Instruct` | TS → matplotlib image, VL model | `run_single_task_gpu.sh` |

---

## Results

*To be filled after `tse_rq1_official_icl` runs complete.*

| Model | Zero-Shot Bal. Acc | ICL k=1 Bal. Acc | Δ | Wilcoxon p |
|---|---|---|---|---|
| `Qwen3-8B-Instruct` | — | — | — | — |
| `ChatTS-8B` | — | — | — | — |
| `Qwen3-VL-8B-Instruct` | — | — | — | — |

---

## Running the Experiment

```bash
# Smoke test — both conditions on tid=1, text model, 5 samples
python run_icl.py --task_id icl_tse_1 --method Qwen/Qwen3-8B-Instruct \
  --prompt_format tse_official --num_shots 0 \
  --num_samples 5 --random_seed 0 --display_samples 3 \
  --tse_data_path qa_dataset_augmented.json

python run_icl.py --task_id icl_tse_1 --method Qwen/Qwen3-8B-Instruct \
  --prompt_format tse_official --num_shots 1 \
  --num_samples 5 --random_seed 0 --display_samples 3 \
  --tse_data_path qa_dataset_augmented.json

# Submit all jobs (~1,764 SLURM jobs)
bash experiment_scripts/icl_tse_rq1_official_icl.sh
```

W&B project: `aviramom-/ts-icl` · exp_id: `tse_rq1_official_icl`
