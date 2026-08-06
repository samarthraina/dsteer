# Reproducibility

This document tracks how to reproduce results in the paper. It will be filled in as scripts are built and experiments run.

## Models

The model pairs used in this work. Each pair shares the same base model and was trained with the same instruction-tuning and DPO recipes.

| Pair name      | Base model                              | IT data         | DPO data              | HF path                                                  |
|----------------|-----------------------------------------|-----------------|-----------------------|----------------------------------------------------------|
| llama3-oh      | meta-llama/Meta-Llama-3-8B              | OpenHermes-2.5  | HH-RLHF harmless-base | sirius5005/SFT-and-DPO                                   |
| llama3p1-oh    | meta-llama/Meta-Llama-3.1-8B            | OpenHermes-2.5  | HH-RLHF harmless-base | divyajot5005/Instruction-Tuning-and-DPO-Models           |
| ministral-oh   | (mistralai Ministral 8B; verify exact)  | OpenHermes-2.5  | HH-RLHF harmless-base | divyajot5005/Instruction-Tuning-and-DPO-Models           |
| qwen25-oh      | (Qwen Qwen2.5 7B or 8B; verify exact)   | OpenHermes-2.5  | HH-RLHF harmless-base | divyajot5005/Instruction-Tuning-and-DPO-Models           |

Confirm exact base model versions with the training repository.

## Training hyperparameters

Read from the training scripts published alongside the checkpoints
(`divyajot5005/Instruction-Tuning-and-DPO-Models`: `dpo.py`, `qwen_dpo.py`).
These supersede the earlier values recorded here, which listed the DPO LR as 5e-7.

| Stage | Pair            | LoRA r | alpha | dropout | LR   | Beta | Epochs | Eff. batch |
|-------|-----------------|--------|-------|---------|------|------|--------|------------|
| DPO   | llama3, llama3p1| 16     | 32    | 0.05    | 1e-4 | 0.1  | 1      | 16 (2x8)   |
| DPO   | qwen2p5         | 16     | 32    | 0.05    | 1e-5 | 0.1  | 1      | 16 (2x8)   |

LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
DPO data: `Anthropic/hh-rlhf`, `harmless-base`, 5% held out for eval.
Reference policy: the merged SFT model, obtained by disabling the DPO adapter.

**Known problems with these runs — see `DECISIONS.md`:**

- The DPO learning rates are 1e-4 and 1e-5. Typical DPO is 5e-7 to 5e-6, so both are
  one to two orders of magnitude high. `qwen_dpo.py` carries the comment
  "more typical for DPO than 1e-4", so the discrepancy was noticed mid-project.
  The llama3p1 checkpoints show severe degenerate repetition consistent with this.
- Training and merging ran in fp16 (`torch_dtype=torch.float16`; merged configs
  declare `"dtype": "float16"`), against the bf16 decision recorded in `DECISIONS.md`.
- A pad token is added and `resize_token_embeddings` is called, giving
  `vocab_size: 128257` — one randomly initialised embedding row.
- Merged llama3p1 configs set `eos_token_id: 128001` (`<|end_of_text|>`) while the
  Llama-3 chat template emits `<|eot_id|>` (128009), so generation does not stop on
  the token the model actually produces.

**Unresolved:** the merged llama3p1 config has `max_position_embeddings: 8192` and
`rope_type: "default"`, which match Llama-3-8B rather than Llama-3.1-8B (131072,
`rope_type: "llama3"`). The base model for this pair needs confirming before use.

## Hardware

To be filled in as we run experiments. Track: GPU model, VRAM, instance provider, approximate runtime per script.

## Random seeds

All scripts use seed 42 unless otherwise specified. Set in:
- Python `random` module
- NumPy
- PyTorch (CPU and CUDA)
- HuggingFace datasets (shuffle operations)
