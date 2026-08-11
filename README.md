# dsteer

How much of what DPO does to a model can be reproduced at inference time by adding a
single direction to the residual stream?

A steering vector read off a pair of checkpoints — one before preference optimisation,
one after — can be added back at generation time without retraining. This repository
measures what that recovers, in both directions: how much of an aligned model's behaviour
can be installed on its predecessor, and how much can be removed from the aligned model
itself.

**Status:** active. Numbers move; the code is the stable part.

## What is here

    src/steering/     the package: activations, steering, generation, judging, figures
    scripts/          one entry point per experiment phase
    scripts/pipelines/ the shell orchestration each multi-stage run was launched with
    configs/          model and experiment configuration
    training/         post-training recipe for the checkpoint pairs
    tests/            guards on the intervention and the scoring
    archive/          the previous version, kept for reference

A note on naming: configs and output paths use `it` for the checkpoint before preference
optimisation. It is the SFT checkpoint. The paths predate settling on the term and
renaming them would orphan every existing result directory.

## Results

Generations, judged scores and activation tensors live in a companion dataset:

    https://huggingface.co/datasets/samarthraina/dsteer-results

`activations.pt` is ~2 GB per pair and is excluded from the run sync, so it is uploaded
separately by `scripts/upload_activations.py`. With those files the geometry analyses
rerun on a laptop — no GPU and no model weights required.

    python scripts/fetch_results.py --restore-sweeps

    activations/   paired residual streams, per pair and readout position
    vectors/       steering directions already derived from those
    runs/          generations, judged scores, ceiling searches, figures

## Two interventions, and why the distinction matters

A direction can be **added** to the residual stream or **ablated** from it, and the two are
not versions of the same thing. Addition moves the state by a fixed multiple of the vector.
Ablation projects the component along it out, so how much is removed depends on what each
activation carries, and the model cannot represent that direction anywhere it is applied.

`--mode add` and `--mode ablate` select between them. One consequence is easy to get wrong:
the sweep negates lambda on the DPO side so that addition steers back toward the earlier
checkpoint, but ablation has no direction to reverse — a negative coefficient *restores*
the component instead of removing more of it. The sign is therefore applied only to
addition, and `tests/test_steering.py` pins both halves.

## Before renting a GPU

    bash scripts/smoke_test.sh /tmp/smoke

Runs every entry point once on a tiny random model, in a couple of minutes, with no GPU.
Every failure this has caught was invisible to the unit tests and to reading the code: a
missing tokenizer dependency, a run directory that did not record which vector produced it
and so resumed on an earlier run's generations, and an intervention applied with the wrong
sign. Run it first.

## Pipeline

Steps 1–3 need a GPU and no judge; step 4 needs the judge and no GPU work, and the two
cannot share a card.

**1. Geometry.** Paired activations for a checkpoint pair, then what they look like.

    python scripts/layer_profile.py --model-config configs/tulu3.yaml \
        --eval-config configs/layer_profile_response_token.yaml --sync
    python scripts/rank_analysis.py   --run tulu3=outputs/layer_profile_response_token/tulu3 ...
    python scripts/vector_analysis.py --run tulu3=outputs/layer_profile_response_token/tulu3 ...

Read position matters. `layer_profile.yaml` reads the final prompt token and
`layer_profile_response_token.yaml` the final token of prompt plus chosen response; the
two give different numbers, so results from one are not comparable to the other.

**2. Steering range.** How hard this model can be pushed, in this direction, before its
generations fall apart. Needs no judge, because degeneration is detectable locally.

    python scripts/lambda_range.py --model-config configs/tulu3.yaml \
        --eval-config configs/steer_sweep.yaml --side it --n-probe 200 \
        --output-dir outputs/lambda_range

Ceilings differ by model and by direction — measured here from 0.30 to 1.15 — so a shared
grid applies unequal intervention across models and runs past the cliff on some of them.

**3. Sweep.** Generate across the range the search found.

    python scripts/steer_sweep.py --model-config configs/tulu3.yaml \
        --eval-config configs/steer_sweep.yaml --side it --sync

**4. Score.**

    python scripts/start_judge_server.py                       # terminal 1
    python scripts/score_sweep.py --sweep-dir outputs/steer/tulu3/harmfulqa/it \
        --it-baseline .../it/baseline.jsonl --dpo-baseline .../dpo/baseline.jsonl

`--recheck METRIC` re-judges a single metric when its rubric changes, rather than paying
for all five. `scripts/screen_model.py` generates from one checkpoint, for deciding
whether a pair is worth building before training one.

**Other entry points.** `refusal_direction.py` builds a direction from harmful-against-
harmless prompt contrasts on a single checkpoint, no pair required, and reports its angle
to the checkpoint-difference vector. `select_refusal_direction.py` picks among per-layer
candidates on a validation split. `train_steering_vector.py` optimises a vector directly
under the DPO loss with the model frozen. `screen_model.py` generates from one checkpoint,
for deciding whether a pair is worth building before training one.

**Figures.**

    python scripts/make_figures.py --runs-root outputs --output-dir outputs/figures

**5. Analysis.** A mean over a whole harmful-prompt benchmark averages the informative
prompts into the ones both checkpoints handle identically, and reports a diluted effect as
a null. Conditioning on the prompts that carry a difference is what makes a transfer number
mean anything, and the subset is fixed by the two unsteered checkpoints alone, so adding an
arm cannot move the set it is judged on.

    python scripts/conditioned_analysis.py         --it-scored  outputs/.../it/scored/baseline_scored.jsonl         --dpo-scored outputs/.../dpo/scored/baseline_scored.jsonl         --arm "checkpoint=outputs/.../dpo_ho300::outputs/.../dpo_ho300_random"         --arm "refusal=outputs/.../dpo_refusal::outputs/.../dpo_refusal_random"         --compare "checkpoint,refusal"

Report `--compare`, not the overlap between two arms' own intervals. Two arms scored on the
same prompts share their baseline, their floor and most of their noise, so the interval on
the difference is far tighter; and overlap between single-arm intervals implies nothing
either way, while non-overlap implies a difference.

**6. Geometry of a direction.** How much of a direction the activations already carry,
which is what ablation can act on and addition is indifferent to. No GPU, no judge.

    python scripts/direction_dispersion.py         --checkpoint-acts outputs/layer_profile_harmfulqa/<model>/activations.pt         --refusal-acts    outputs/refusal_direction/<model>/activations.pt

Projections are reported as a share of activation norm, because the raw inner product says
nothing without knowing how large the activation is.

## Things that are easy to get wrong

Each of these changes a reported number without producing an error, and each is handled
explicitly in the code rather than left to the caller.

- `hidden_states[-1]` is the output of the final norm, not the last residual stream layer.
  Activations are captured by forward hook, and the post-norm output is kept separate.
- A judge asked for an integer at temperature 0 concentrates on a few anchors, so an
  eleven-point scale can behave like a three-point one. Scores are probability-weighted,
  and the raw integer is kept beside the weighted value.
- Over-steering produces empty strings and repetition loops. Those are screened before
  judging; averaging them in flatters exactly the region where the model has collapsed.
- Judge abstentions are counted, never coded as zero — in the de-alignment direction the
  outputs a judge declines to score are the unsafe ones.
- Harmfulness is judged with the request in view. Without it the judge reads an answer
  with no idea what was asked, which under-scores terse harmful replies by about a fifth.

**A steering strength belongs to a vector, not to a checkpoint.** Ceilings measured for one
construction do not transfer to another built from a different corpus, readout position or
hold-out. Reading an arm at a borrowed magnitude degenerated a third of its generations in
one case here and nine tenths in another. Worse, the reported number *improves* as the model
degrades, because collapsed generations carry no score and leave the numerator while the gap
they are divided by comes from every prompt. Measure the ceiling first; it needs no judge.

**A row index means a different prompt in different activation files.** The loaders shuffle
on a fixed seed and take the first n, so files built with different hold-outs are offset
relative to each other. Comparing them row for row gives roughly the similarity of two
unrelated prompts, which reads like a difference in extraction rather than in alignment.

## Setup

    git clone https://github.com/samarthraina/dsteer.git
    cd dsteer
    pip install -r requirements.txt
    huggingface-cli login          # or export HF_TOKEN

The judge server pins its own torch build, so it wants a separate environment from the
generation code.

    pytest tests/ -v               # CPU only, pulls a tiny model on first run

## License

MIT
