# D-STEER: Preference Alignment Techniques Learn to Behave, not to Believe — Beneath the Surface, DPO as Steering Vector Perturbation in Activation Space

> **The Illusion of Alignment:** Direct Preference Optimization (DPO) aligns language models not by restructuring semantics, but through low-rank vector steering in activation space.

This repository contains the official implementation for **D-STEER**. We demonstrate that the safety and helpfulness behaviors induced by DPO can be replicated—and controlled—by extracting a static "alignment vector" and injecting it into the forward pass of a base SFT model.

---

## 📖 Table of Contents

* Abstract
* Methodology
* Usage Pipeline
* Evaluation & Key Findings
* Universal Steering Shift
* The Optimal Range Limit
* Adversarial Asymmetry
* The Refusal Paradox

* Repository Structure

---

## <a id="abstract"></a>🧩 Abstract

Recent alignment techniques like DPO represent a significant leap in model safety. However, the geometric nature of this alignment remains under-explored. In this work, we show that DPO-aligned models differ from their SFT baselines primarily along specific linear directions in activation space.

By extracting these directions, we introduce **D-STEER**, a method to linearly interpolate between SFT and DPO behaviors at inference time. We validate this approach via **126,000 metric evaluations** across three benchmarks, proving that a simple activation offset can replicate complex safety behaviors, contextual refusals, and coherence improvements without full model retraining.

---

## <a id="methodology"></a>🧪 Methodology

Our approach relies on the extraction of a **Steering Vector** () that captures the mean difference in representation between the SFT model () and the DPO model ().

### Algorithm: Activation Steering

Given a preference dataset , we extract hidden states  at layer  for the final token:

1. **Compute Activation Difference:**


2. **Extract Mean Steering Vector:**


3. **Inference-Time Injection:**
During the forward pass of the SFT model, we modify the hidden states using a steering coefficient :


* λ > 0 : Steers SFT -> DPO (Aligns the model).
* λ < 0: Steers DPO -> SFT (Removes alignment).



---

## <a id="usage"></a>🚀 Usage Pipeline

### 1. Vector Extraction & Steering

The core logic for extracting activation differences and generating steered responses is contained in the optimized notebook.

* **Notebook:** `Create_steering_vector_and_AQI_eval.ipynb`
* **Action:** Open this notebook to load your base/DPO models, compute the vector , and generate outputs across a range of  values.

### 2. Automated Evaluation

We provide a highly parallelized evaluation pipeline using `vLLM` for generation and `G-Eval` for scoring.

**Step A: Initialize Judge Server**
Start the local OpenAI-compatible API server using `vLLM` (hosting Qwen2.5-32B-Instruct as the judge).

```bash
python server.py

```

*> **Note:** Keep this terminal window open.*

**Step B: Run Evaluation Metrics**
Execute the batch evaluation script. This will assess outputs on Relevance, Toxicity, Helpfulness, and Steering Shift.

```bash
python evaluation.py

```

---

## <a id="evaluation"></a>📊 Evaluation & Key Findings

We conducted an extensive evaluation on **1,000 prompts** across three benchmarks: **HH-RLHF** (Conversational), **HarmfulQA** (Adversarial), and **AdvBench** (Optimized Jailbreaks).

### <a id="steering-shift"></a>1. Universal Steering Shift

We introduce the **Steering Shift** metric to quantify behavioral alignment. Across all datasets, steering provides a robust signal with a consistent dynamic range (≈0.79).

* **HarmfulQA:** 0.792 Range
* **HH-RLHF:** 0.797 Range
* **AdvBench:** 0.765 Range

Positive steering (λ > 0) successfully interpolates behavior from SFT-like (0.10) to DPO-like (0.90) regardless of the prompt distribution.

### <a id="optimal-range"></a>2. The Optimal Range Limit

We identified a fundamental architectural limit on steering magnitude.

* **Safe Range:** 
* **Collapse:** Beyond , response quality degrades catastrophically. At , we observe **repetition loops** (e.g., 36.3% 4-gram repetition on HH-RLHF) and a 51% drop in quality scores.

### <a id="asymmetry"></a>3. Adversarial Asymmetry

A novel finding on the **AdvBench** (Jailbreak) dataset:

* **Adding Alignment (λ > 0):** Smooth, monotonic improvement.
* **Removing Alignment (λ < 0):** Chaotic, non-monotonic oscillation.
This suggests that removing alignment from a model trained against optimized attacks destabilizes its learned defenses unpredictably, whereas adding alignment is geometrically stable.

### <a id="refusal-paradox"></a>4. The Refusal Paradox

On **HH-RLHF** and **HarmfulQA**, the unaligned SFT model often has a *higher* refusal rate than the DPO model, yet scores lower on helpfulness.

* **SFT Behavior:** Blunt, generic refusals (Avg 268 chars).
* **DPO/Steered Behavior:** Nuanced, educational refusals (Avg 1,153 chars).
* **Conclusion:** DPO's value is not just "safety" (which is already high in SFT), but **behavioral refinement**—providing context and explanations.

---

## <a id="structure"></a>📂 Repository Structure

| File | Description |
| --- | --- |
| `Create_steering_vector_and_AQI_eval.ipynb` | **Core Logic:** Vector extraction, layer selection, and steering inference. |
| `evaluation.py` | **Metrics:** Automated G-Eval pipeline (Relevance, Quality, Toxicity, etc.). |
| `server.py` | **Inference:** vLLM server setup for the Judge model. |
| `requirements.txt` | **Deps:** Python package requirements. |
| `index.html` | **Demo:** Frontend visualization for steering interpolation. |
