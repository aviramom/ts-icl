# In-Context Learning for Time Series LLM (TSLM) Tasks

> Foundation document for the TS-ICL project. We study in-context learning (ICL) for a
> multimodal Time Series LLM — a model that takes time series + text as input and generates
> text (captions, QA, forecasts, class labels). The TSLM injects time-series patch embeddings
> into a (LoRA-tuned) instruction LLM. ICL here means *adapting to a task purely from in-context
> demonstrations supplied at inference, without any weight updates.*

---

## 1. Introduction

Large (multimodal) LLMs can solve new tasks by conditioning on a few input–output demonstrations
placed in the prompt — *in-context learning* — without gradient updates. For text-only LLMs this
is well established; for vision-language models (VLLMs) it is fragile and task-dependent. For
**time series**, ICL is largely unexplored: demonstrations carry both a numeric/temporal signal
and accompanying text, the signal is injected as many patch tokens (expensive in context length),
and similarity between examples is itself a non-trivial, multimodal notion. This project asks
whether and how ICL helps a TSLM, guided by three research questions.

**RQ1 — Does ICL help in TSLM tasks?**
Do in-context `(time series, text/label)` demonstrations improve performance over zero-shot across
the TSLM task families — forecasting, captioning/description, QA, and classification? Under what
conditions does it help: which task types benefit, how does demonstration *quality* and label
correctness matter, how does the effect scale with model size and with the number of shots `k`,
and when (per the multimodal-ICL literature) does adding shots *hurt*?

**RQ2 — How to retrieve relevant context shots across modalities?**
Given a pool of candidate demonstrations whose keys mix **text** and **time series**, how do we
select the most useful shots for a test query? This spans TS-based similarity (DTW, shape/frequency
features, learned-embedding cosine), text-semantic similarity, and **joint/fused** retrieval over
both modalities — plus the *ordering* of the selected shots (recency/position effects) and
robustness to irrelevant or noisy shots.

**RQ3 — Long-context implications and effectiveness.**
A single time series consumes many tokens (≈ `length / patch_size` patch embeddings), so shot count
and per-shot resolution directly inflate context length. How does accuracy trade off against
context length and cost? Does utility grow monotonically with `k`, where does it **saturate or
degrade**, and how sensitive is the model to *where* in the context the relevant shot sits
(position/recency bias)?

**Our approach.** To answer these questions empirically, we build a **TS-ICL benchmark** composed of
several datasets spanning the TSLM task families (forecasting, captioning/description, QA, and
classification). Each dataset is organized into a *query* set and a *candidate pool* of
demonstrations so that retrieval (§2.2) and shot-count / long-context sweeps (§2.3) can be run under
controlled conditions, with permutation-augmented variants to measure ordering and noise
sensitivity. The benchmark lets us hold the TSLM fixed and vary only the in-context set `S`,
isolating the effect of ICL itself.

> **TODO — benchmark composition.** Enumerate the datasets, their tasks/domains, sizes, the
> query/pool split protocol, the per-task utility metric `U`, and the number of permutation/noise
> variants per problem. _(to be filled in as datasets are finalized)_

---

## 2. Problem Formulation

### 2.1 In-Context Learning for TSLM

We adapt the standard VLLM-ICL definition to the time-series modality.

> *(Reference, VLLM-ICL):* Given a pre-trained VLLM `θ`, an optional text instruction `I`, a
> context set `S = {(xᵢ, yᵢ)}` of examples `xᵢ` and labels `yᵢ`, and a test example `x*`, ICL
> models estimate `p_θ(y* | x*, I, S)` with a single feed-forward pass. For LLMs `x, y` are text;
> for VLLMs `x` can be text and/or images and `y` can be text or images.

For a **TSLM** `θ`, we generalize the non-text modality *form VLLM-ICL* to time series. Define each
example’s input as a (possibly partial) pair of modalities

```
x = (τ, t),   τ ∈ ℝ^{C×L}  (C channels, length L; optional),   t = text (optional)
```

and the label `y` as text (forecast-as-text, caption, QA answer, class label) or a numeric/temporal
target. Given `θ`, an optional instruction `I`, an ordered context set of `k` demonstrations

```
S = ((x₁, y₁), …, (x_k, y_k)),
```

and a test input `x* = (τ*, t*)`, the TSLM estimates

```
        p_θ ( y* | x*, I, S )                                      (1)
```

in one feed-forward pass. Internally each `τ` is mapped to patch embeddings and spliced into the
token stream at `<|ts_temp|>` positions (the repo’s injection mechanism). This recovers the
text-only LLM case (`τ = ∅`) and the VLLM case (image instead of `τ`) as special instances; the
distinguishing feature here is that the non-text modality is a **continuous, variable-length,
multi-channel time series injected as patch tokens.**

The ICL objective is to maximize expected task utility over a distribution of tasks/queries:

```
        max_{S}  𝔼_{x*}[ U( y*, ŷ ) ],   ŷ ~ p_θ( · | x*, I, S )      (2)
```

where `U` is a task-appropriate score (negative MAE/MSE/MASE for forecasting; accuracy for
QA/classification; caption-quality metrics for description). Two control knobs define the rest of
the formulation: **which** demonstrations populate `S` (retrieval, §2.2) and **how large** `S` may
be (long-context budget, §2.3).

### 2.2 The Retrieval Problem

In practice `S` is not given but **selected** from a large candidate pool. Let

```
P = {(x_j, y_j)}_{j=1}^{N}
```

be the pool. A retriever `R` maps a test query and pool to an *ordered* subset of size `k`:

```
        S = R(x*, P, k),   S ⊆ P,   |S| = k.                        (3)
```

The retrieval objective mirrors (2): choose `R` (equivalently choose `S`) to maximize expected
utility of the resulting prediction,

```
        max_{S ⊆ P, |S|=k}  𝔼[ U( y*, ŷ ) ],   ŷ ~ p_θ( · | x*, I, S ).   (4)
```

A tractable surrogate ranks pool items by a **multimodal similarity** to the query and takes the
top-`k`:

```
        s(x*, x_j) = α · s_ts(τ*, τ_j) + (1 − α) · s_text(t*, t_j),   α ∈ [0,1]   (5)
```

where `s_ts` is a time-series similarity (e.g. DTW, shape/frequency-feature distance, or cosine
between learned TS-encoder embeddings) and `s_text` a text-semantic similarity (e.g. cosine between
sentence embeddings). `α` interpolates between TS-only and text-only retrieval; a learned **joint
embedding** `φ(τ, t)` with `s = cos(φ(x*), φ(x_j))` is an alternative to the linear fusion in (5).
Two properties matter beyond top-`k` scoring: (i) **ordering** — `S` is a sequence, and the
permutation of selected shots affects `p_θ` (recency/position bias); and (ii) **robustness** — a
single irrelevant/noisy shot can degrade performance, so `R` must be selective, not merely
high-recall.

### 2.3 The Long-Context Problem

Each demonstration costs context tokens, and time series are token-heavy. For a series of length
`L` with patch size `p`, the TS contributes roughly `L/p` patch tokens. The total prompt length for
context set `S` is

```
        T(S) = |I| + Σ_{i=1}^{k} ( |x_i| + |y_i| ) + |x*|,
        with  |x_i| ≈ (L_i / p) + |t_i|.                            (6)
```

Given a hard budget `T_max` (model context window, latency, or cost ceiling), ICL becomes a
**budget-constrained selection** problem:

```
        max_{S}  𝔼[ U(y*, ŷ) ]   s.t.   T(S) ≤ T_max.               (7)
```

This exposes a three-way tradeoff among the **number of shots `k`**, the **per-shot TS resolution**
(downsampling / larger `p` shrinks `L/p` but loses detail), and the **budget**. The effectiveness
questions follow directly: is `U` monotonically increasing in `k`, or does it **saturate / degrade**
past some `k*`? How is utility distributed over shot *position* (does the model attend mostly to the
most recent shots)? And what is the accuracy-per-token (or accuracy-per-dollar) frontier that an
effective retriever + budget policy should ride?

---

## 3. Methods

### 3.1 Overview

We hold the TSLM `θ` fixed and study three levers from §2: (a) **prompt assembly** — how the
ordered context set `S` and instruction `I` are serialized into the token stream (TS injected as
patch tokens at `<|ts_temp|>` positions, interleaved with each demonstration’s text/label); (b)
**retrieval** — which candidates from the pool `P` populate `S` (§3.2); and (c) **budget/shot-count
policy** — how many shots and at what per-shot resolution we spend the context budget `T_max` (§2.3).
Below we focus on the retrieval mechanism, which is the core of RQ2.

### 3.2 Embedding-Based Retrieval

The retriever embeds the query and each pool item into a shared space and ranks by cosine
similarity, returning the top-`k` (Eq. 3–5). Given a query embedding and a database of pool
embeddings, top-`k` selection is a normalized dot-product:

```python
import torch
import torch.nn.functional as F

# Assume query_emb is [1, 768] and database_embs is [N, 768]
# 1. Normalize the embeddings to use dot-product as cosine similarity
query_emb = F.normalize(query_emb, p=2, dim=-1)
database_embs = F.normalize(database_embs, p=2, dim=-1)

# 2. Compute similarity matrix
similarities = torch.mm(query_emb, database_embs.T)  # Shape: [1, N]

# 3. Get top K results
topk_scores, topk_indices = torch.topk(similarities, k=5, dim=-1)
```

This is the engine behind the surrogate scoring `s(x*, x_j)` of Eq. 5: the only design choice is
**what produces the embeddings** (and, for fused retrieval, how the per-modality scores are
combined via `α`). The `database_embs` are precomputed once for the pool `P` and cached, so
retrieval at inference is a single matvec plus a top-`k`. Two practical notes: (i) `topk_indices`
returns items in descending similarity — since ordering of `S` matters (recency/position bias,
§2.2), we treat the returned order as a *policy* to ablate (e.g. most-similar-last vs.
most-similar-first); (ii) the same routine supports **fused** retrieval by ranking on
`α·s_ts + (1−α)·s_text` instead of a single embedding space, or by concatenating/normalizing the two
embedding blocks before the matvec.

### 3.3 Preliminary Embedding / Similarity Approaches

We will compare several ways to produce `s_ts` (the time-series side of Eq. 5); the text side
`s_text` uses standard sentence embeddings. Candidate TS approaches, roughly from
signal-level to model-level:

- **Euclidean / L2 distance** on aligned raw (z-normalized) series — simplest baseline; sensitive to
  phase shift and length, but cheap and a useful lower bound.
- **Dynamic Time Warping (DTW)** — shape-based similarity robust to local time shifts/warping;
  classic for TS retrieval. More expensive (`O(L²)` per pair) so used as a strong reference or on
  downsampled series.
- **Spectral similarity / correlation via FFT** — compare power spectra or cross-correlation in the
  frequency domain; captures periodicity/seasonality matches that L2 misses, and is shift-invariant.
- **Distributional / statistical-divergence measures** — treat a series (or a window) as a sample
  from a distribution and compare distributions rather than aligned points: **Wasserstein /
  optimal-transport distance**, **kernel Maximum Mean Discrepancy (MMD)**, and energy/distance
  metrics (e.g. distance correlation, Cramér distance). These are reordering-/phase-robust and
  capture differences in amplitude/value distribution and higher moments; MMD also reduces to a
  kernel inner product, so it slots naturally into the embedding/kernel view of §3.2.
- **Learned TS-encoder embeddings** — cosine similarity between embeddings from the TSLM’s own TS
  encoder (the patch-embedding output), so retrieval lives in the same space the model consumes.
- **Vision-model embeddings of TS-as-image** — render each series as a plot and embed with a
  pretrained vision encoder (e.g. **DINOv2**, **SigLIP**, or a **Qwen-VL** vision tower), then use
  cosine similarity. Leverages strong visual-pattern priors and connects to the multimodal-ICL
  literature in §4.

Each approach yields a `database_embs` (or a pairwise distance reducible to a kernel) that plugs
directly into §3.2. We evaluate them by downstream task utility `U` (Eq. 4), not just retrieval
recall, since the goal is the shots that most help the TSLM — not merely the most "similar" ones.

---

## 4. Related Work

| Title (link) | Year / Venue | Modality | Problem Addressed | Key Findings | Relevance to Us |
|---|---|---|---|---|---|
| [In-Context and Few-Shot Learning for Forecasting Time Series Data based on LLMs](https://arxiv.org/html/2512.07705v1) | 2025 (arXiv 2512.07705) | Time series → numeric (as text) | Can general LLMs (o4-mini, Gemini 2.5 Flash Lite) with zero-/few-shot prompts beat a TS foundation model (TimesFM) and DL baselines (LSTM, TCN) on the SWaT water-treatment forecasting task, without fine-tuning? | TimesFM best (RMSE 0.3025, fast). o4-mini competitive in zero-shot (RMSE 0.3310) but much slower; few-shot prompting helps modestly; classic DL underperforms — foundation models exploit pretrained temporal priors. | Directly probes **RQ1** for forecasting and frames the ICL-LLM vs. TS-foundation-model baseline comparison we should include. |
| [SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](https://arxiv.org/abs/2506.21355) | 2025 (arXiv 2506.21355) | Image + text (medical) | First benchmark for multimodal *medical* ICL: 111 expert-curated problems (SMMILE++ → 1,038 permutations), 6 specialties, 13 imaging modalities, 517 image-question-answer triplets. | ICL gives only ~8% average gain over zero-shot; a single noisy/irrelevant shot drops performance up to 9.5%; strong **recency bias** — good shot ordering yields up to +71%. | Motivates **RQ2** (noise robustness of retrieval) and **RQ3** (ordering/recency); a template for expert-driven, permutation-augmented benchmark design. |
| [VL-ICL Bench: The Devil in the Details of Multimodal ICL](https://arxiv.org/abs/2403.13164) | 2024 (arXiv 2403.13164) | Image + text (in & out) | Argues prior multimodal-ICL evaluation (VQA, captioning) is too narrow and tests neither the strengths nor limits of ICL; introduces a broad task suite spanning perception → reasoning and **long context length**. | Even GPT-4-class models struggle; the true capabilities and limitations of multimodal ICL remain under-explored — details (task design, shot format) dominate measured performance. | Methodology blueprint for designing a **TS-ICL task suite** beyond captioning, and a precedent for explicit long-context stress tests (**RQ3**). |
| [FewMMBench: A Benchmark for Multimodal Few-Shot Learning](https://arxiv.org/abs/2602.21854) | 2026 (arXiv 2602.21854) | Image + text | Systematic few-shot evaluation of ICL + chain-of-thought across diverse multimodal tasks (attribute recognition → temporal reasoning); 26 open-weight models across 6 families. | Instruction-tuned models are strong zero-shot but **benefit minimally — or regress —** with more demonstrations or CoT; retrieval-based demos and larger context windows give limited gains. | Cautionary evidence for **RQ1/RQ3**: we must isolate *when* ICL actually helps; its retrieval-demo limitations sharpen **RQ2** (why naive retrieval underdelivers). |

**Takeaway.** Across image-centric multimodal benchmarks, ICL gains are repeatedly found to be
**small, fragile, and highly sensitive to shot ordering and noise**, while the only time-series
entry treats series as text and compares against foundation models rather than studying ICL
mechanics. This leaves a clear gap: a **time-series-native** study of ICL that takes the TS modality
(patch-token injection, multimodal retrieval, long-context cost) seriously — which is the goal of
this project.

---

## 5. Next Steps / Open Questions

- **RQ1:** Establish zero-shot vs. `k`-shot baselines on each TSLM task family (forecasting,
  captioning, QA, classification); measure utility vs. `k` and vs. model scale.
- **RQ2:** Implement and compare retrievers — TS-only (`s_ts`), text-only (`s_text`), linear fusion
  (Eq. 5), and a learned joint embedding `φ(τ, t)`; ablate shot ordering and noisy-shot robustness.
- **RQ3:** Map the accuracy-vs-context-length frontier; find the saturation point `k*` and quantify
  position/recency bias; study the `k` ↔ per-shot resolution tradeoff under a fixed `T_max`.
- **Benchmark:** Decide whether to adopt/extend an existing suite (VL-ICL Bench / FewMMBench style)
  or curate a TS-native benchmark with controlled, permutation-augmented problems (SMMILE style).
