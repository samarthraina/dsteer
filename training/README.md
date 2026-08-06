# Training the checkpoint pairs

The pair the main results rest on is a two-stage post-training of Llama-3-8B:

    meta-llama/Meta-Llama-3-8B
      -> SFT   on OpenHermes-2.5           (llama3_SFT.py)
      -> DPO   on HH-RLHF harmless-base    (dpo.py)
      -> merge adapters into the base      (merge.py)

Both stages are LoRA (r=16, alpha=32, dropout 0.05) over the seven attention and MLP
projections, merged into the base weights afterwards so the released checkpoints are
plain causal LMs with no adapter files.

The DPO stage matters for how the results read: it trains on *harmless-base only*, so it
is a safety-specific stage rather than a general preference mix. That is why this pair has
a measurable refusal and harmfulness difference between its two checkpoints where the
others do not — most public post-training installs safety during SFT and uses the
preference stage for helpfulness and style.

## Hyperparameters, as they appear in the scripts

| | SFT | DPO |
|---|---|---|
| data | OpenHermes-2.5 | HH-RLHF `harmless-base` |
| epochs | 1 | 1 |
| learning rate | 1e-4 | 1e-4 |
| batch / accumulation | 4 x 4 = 16 | 2 x 8 = 16 |
| max length | `MAX_SEQ_LENGTH` | 1024 |
| beta | — | 0.1 |
| schedule | cosine, 5% warmup | cosine, 5% warmup |
| precision | **fp16** | **bf16** |
| seed | 42 | 42 |

Two things worth stating plainly rather than leaving for a reader to find.

**The learning rate is 1e-4 for both stages.** That is high for DPO, where 5e-7 to 5e-6 is
the usual range. It is what was run, and the pair behaves sensibly, but it is not a
conventional setting and any comparison to other DPO checkpoints should account for it.

**The two stages use different precision.** SFT runs in fp16 with bf16 disabled; DPO runs
in bf16. The SFT script notes fp16 as the choice for pre-Ampere hardware. Mixed precision
across stages is not ideal for a controlled comparison, though the displacement we measure
is far larger than any plausible precision artefact.

## Provenance

These scripts come from `divyajot5005/Instruction-Tuning-and-DPO-Models` and describe the
Llama-3-8B / OpenHermes / harmless-base recipe. The checkpoints actually used here are
published at `sirius5005/SFT-and-DPO`. The recipe matches, but the scripts have not been
re-run end to end to confirm they reproduce those exact weights bit for bit — so treat
them as the documented procedure rather than as verified provenance. Re-running is the
only way to settle it, and it costs roughly six GPU-hours per stage.

`Qwen2.5_SFT.py` and `qwen_dpo.py` are the same recipe for a Qwen2.5 pair, kept for
completeness. That pair is not used in the results: its instruction-tuned checkpoint
emits a spurious assistant turn mid-output and its DPO checkpoint loops, so neither is
sound to measure.

## Fetching a checkpoint

The DPO repository carries an adapter at its root whose `base_model_name_or_path` points
at a path on the trainer's machine. PEFT follows that regardless of which subfolder is
requested, so loading the repo directly fails. Fetch the two merged subfolders instead:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sirius5005/SFT-and-DPO",
                  allow_patterns=["SFT_merged/*", "DPO_merged/*"],
                  local_dir="models/llama3-oh")
```

`configs/llama3_oh_local.yaml` expects that layout.
