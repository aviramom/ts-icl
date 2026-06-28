# TSE Prompt-Format Ablation

**Experiment:** `tse_prompt_format_ablation_qwenvl` (TSE, ~98 templates)  
**Setup:** k=1 · random picking · 3 seeds (0, 3, 6) · Qwen3-VL-8B-Instruct only  
**Arg:** `--prompt_format {no_support, desc_first, no_desc, desc_last}`

---

## Motivation

The previous `tse_desc_ablation_full` experiment showed that adding the question text (`use_label_desc=1`) has no significant effect on QwenVL's accuracy on TSE (Δ = +0.0011, p = 0.746). Two hypotheses remain unexplored:

1. **Does the description help at all when there are no examples?**  
   If QwenVL can answer TS questions zero-shot from the question text alone, that tells us the question is informative but redundant when examples are present.

2. **Does description *position* matter?**  
   The prior experiment always placed the question *before* the examples (`desc_first`). Transformer attention decays with distance — if the question text is meaningful signal, placing it immediately before the query TS (`desc_last`) should produce a stronger effect than placing it at the start of a long context.

This experiment crosses both axes for QwenVL on TSE.

---

## The 4 Conditions

| # | `--prompt_format` | Has examples? | Has question? | Question position |
|---|---|---|---|---|
| 1 | `no_support` | No (zero-shot) | Yes | Before query |
| 2 | `desc_first` | Yes (k=1) | Yes | Before examples |
| 3 | `no_desc` | Yes (k=1) | No | — |
| 4 | `desc_last` | Yes (k=1) | Yes | After examples, before query |

Conditions 2 and 3 replicate the existing ablation conditions (B and A) for direct comparison.

---

## Prompt Samples

All four examples below use **tid=1** (trend type: Linear / Exponential / No Trend), **seed=0**, **k=1** random picking. TS values are truncated to the first 6; full series are 256 values.

---

### Condition 1 — `no_support` · Zero-shot with question

```
Time Series Classification.
Question: What is the type of the trend of the given time series?

Options:
A) Linear
B) Exponential
C) No Trend

--- TARGET ---
New Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]
Return ONLY the label as one of: [A, B, C] without any explanation
```

No examples at all. The model must classify the query purely from the question text and option legend.

---

### Condition 2 — `desc_first` · Question before examples (replicates existing condition B)

```
Time Series Classification.
Question: What is the type of the trend of the given time series?

Options:
A) Linear
B) Exponential
C) No Trend

--- EXAMPLES ---

Example 1 Time Series: [0.2379, -0.3516, -1.0774, -1.0349, -1.1402, 1.4715, ... (256 values total)]
Label: A

Example 2 Time Series: [-0.1419, 0.5779, 1.1156, 0.9655, 1.0936, 1.8459, ... (256 values total)]
Label: B

Example 3 Time Series: [5.2145, 5.1691, 5.1507, 5.0919, 5.1921, 5.1585, ... (256 values total)]
Label: C

--- TARGET ---
New Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]
Return ONLY the label as one of: [A, B, C] without any explanation
```

---

### Condition 3 — `no_desc` · Examples only, no question (replicates existing condition A)

```
Time Series Classification.

--- EXAMPLES ---

Example 1 Time Series: [0.2379, -0.3516, -1.0774, -1.0349, -1.1402, 1.4715, ... (256 values total)]
Label: A

Example 2 Time Series: [-0.1419, 0.5779, 1.1156, 0.9655, 1.0936, 1.8459, ... (256 values total)]
Label: B

Example 3 Time Series: [5.2145, 5.1691, 5.1507, 5.0919, 5.1921, 5.1585, ... (256 values total)]
Label: C

--- TARGET ---
New Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]
Return ONLY the label as one of: [A, B, C] without any explanation
```

---

### Condition 4 — `desc_last` · Question after examples, immediately before query

```
Time Series Classification.

--- EXAMPLES ---

Example 1 Time Series: [0.2379, -0.3516, -1.0774, -1.0349, -1.1402, 1.4715, ... (256 values total)]
Label: A

Example 2 Time Series: [-0.1419, 0.5779, 1.1156, 0.9655, 1.0936, 1.8459, ... (256 values total)]
Label: B

Example 3 Time Series: [5.2145, 5.1691, 5.1507, 5.0919, 5.1921, 5.1585, ... (256 values total)]
Label: C

--- TARGET ---
Question: What is the type of the trend of the given time series?

Options:
A) Linear
B) Exponential
C) No Trend

New Time Series: [-0.4234, 0.1364, 0.1361, 0.4296, -0.1615, -0.1253, ... (256 values total)]
Return ONLY the label as one of: [A, B, C] without any explanation
```

The question is placed inside the `--- TARGET ---` block, immediately before the query TS. This is the recency-bias condition: the model finishes reading the question text and the option labels right before it must generate its answer.

---

## What Each Condition Tests

| Condition | What a positive effect would mean |
|---|---|
| `no_support` > random | QwenVL can read the question and do meaningful zero-shot TS classification |
| `desc_first` ≈ `no_desc` | Question text adds no signal when examples are present (replicates prior finding) |
| `desc_last` > `desc_first` | Recency matters: the question is more useful when close to the query |
| `desc_last` > `no_desc` | Recency is enough to overcome the neutral/negative effect of the early-position description |

---

## Results

*To be filled after `tse_prompt_format_ablation_qwenvl` runs complete.*

| Condition | Macro Bal. Acc | Δ vs `no_desc` | Wilcoxon |
|---|---|---|---|
| `no_support` | — | — | — |
| `desc_first` | — | — | — |
| `no_desc` | — | — | — |
| `desc_last` | — | — | — |

---

## Running the Experiment

```bash
# Generate augmented TSE data if not present
python scripts/generate_tse_augmented.py \
  --tse_repo third_party/TimeSeriesExam \
  --output qa_dataset_augmented.json \
  --num_variants 10

# Submit all jobs (~1,176 SLURM jobs)
bash experiment_scripts/icl_tse_prompt_format_ablation.sh
```

W&B project: `aviramom-/ts-icl` · exp_id: `tse_prompt_format_ablation_qwenvl`
