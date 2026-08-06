# Running experiments

Everything here assumes a rented GPU instance that can disappear. Three rules follow
from that, and the tooling exists to make them cheap:

1. **Long jobs run inside tmux.** A dropped SSH connection kills a foreground process.
2. **Long jobs checkpoint.** A killed process should cost minutes, not hours.
3. **Results go to the hub.** Anything only in `/workspace` dies with the rental.

## Connecting

```bash
ssh vast                                  # see ~/.ssh/config
source /venv/main/bin/activate
cd /workspace/dsteer
export PYTHONPATH=/workspace/dsteer/src
```

Authenticate once per instance, for gated checkpoints and for upload:

```bash
hf auth login          # needs a token with WRITE scope
```

## tmux

Never run a sweep in a bare SSH session.

```bash
tmux new -s profile          # start a named session
# ... launch the job ...
# detach with: Ctrl-b then d

tmux ls                      # list sessions
tmux attach -t profile       # reattach, from any machine
tmux kill-session -t profile # done with it
```

One session per concurrent job, named after it: `profile`, `steer`, `judge`, `dpo`.
Useful when a job is running:

```bash
tmux new -s profile 'python scripts/layer_profile.py ... 2>&1 | tee run.log'
```

so the output survives even if the pane scrollback does not.

## Checkpointing and resume

`layer_profile.py` flushes activations every `checkpoint_every` prompts (default 200)
to `it.partial.pt` / `dpo.partial.pt` in the output directory. Re-running the same
command picks up from the last flush; the partials are deleted once `activations.pt`
is written. Writes are atomic, so a crash mid-flush cannot corrupt the checkpoint.

```bash
python scripts/layer_profile.py --model-config configs/tulu3.yaml \
    --eval-config configs/layer_profile.yaml            # resumes if interrupted
python scripts/layer_profile.py ... --no-resume         # start clean
```

`dpo_eval.py` and `it_eval.py` resume differently, by prompt id: they append to the
output JSONL as they go and skip ids already present. Deleting the JSONL restarts them.

A resumed run and a fresh run are only comparable if the config is unchanged — the
checkpoint refuses to load if the tensor shape no longer matches, but it cannot detect
a changed prompt set at the same size. Use `--no-resume` when the config changed.

## Two environments

Generation and judging do not share an interpreter. vLLM pins its own torch: installing
it beside the generation stack replaced torch 2.12.0+cu126 with 2.11.0+cu130, which
silently invalidates a run already in flight. They communicate over HTTP, so there is no
reason to share.

```bash
/venv/main/bin/python              # extraction, steering, generation, analysis
/workspace/venv_judge/bin/python   # vLLM judge server only
```

Building the judge environment, once per instance:

```bash
python3 -m venv /workspace/venv_judge
/workspace/venv_judge/bin/pip install vllm
```

Scoring is prefill-bound -- every call repeats the same rubric prefix -- so the server
runs with prefix caching, which is the default:

```bash
tmux new -s judge '/workspace/venv_judge/bin/python scripts/start_judge_server.py'
```

Wait for the model to finish loading before starting a scoring run. The judge is bf16,
not quantised: scores are compared against earlier bf16 runs, and quantisation moves
borderline cases, which is exactly where a banded rubric changes its answer.

Do not start the judge while a generation sweep is running. Qwen-32B in bf16 wants
about 64 GB and an 8B generation run about 16 GB, which is the whole card.

## GPU usage

Every run writes `gpu_usage.json` beside its results:

```bash
python scripts/layer_profile.py ... --hourly-rate 1.338
```

```json
{
  "elapsed_hours": 0.42, "estimated_cost_usd": 0.56,
  "gpu_util_mean_pct": 71.4, "gpu_idle_fraction": 0.08,
  "mem_peak_fraction": 0.31, "torch_peak_alloc_mb": 24880.2
}
```

Two of these decide what to do next. **`gpu_idle_fraction`** — samples below 20%
utilisation — is the share of the bill buying nothing; if it is high the job is
bounded by tokenisation, disk, or the judge rather than the GPU, and a bigger card
will not help. **`mem_peak_fraction`** is the headroom: well under 1.0 means the
batch size can go up.

Live view while a job runs, from a second tmux pane:

```bash
nvidia-smi dmon -s um        # utilisation and memory, one line per second
watch -n 5 nvidia-smi
```

## Getting results off the machine

Scripts that support it take `--sync`:

```bash
python scripts/layer_profile.py ... --sync
```

Otherwise, or for a run still in progress:

```bash
python scripts/sync_results.py --dir outputs/layer_profile/tulu3 \
    --experiment layer_profile --model tulu3

# back up every 10 minutes while a long sweep writes
python scripts/sync_results.py --dir outputs/dpo_eval/tulu3 \
    --experiment dpo_eval --model tulu3 --watch 600
```

Layout on the hub:

```
runs/{experiment}/{model}/{run_id}/
```

`run_id` is a UTC timestamp, so re-runs never overwrite each other. Every run
directory carries `run_meta.json` with the git commit, GPU, and package versions that
produced it — which is what makes a number in the paper traceable to a run.

`activations.pt` (~2 GB) is skipped by default; pass `--include-weights` when the raw
tensors are genuinely needed elsewhere.

## Before destroying an instance

```bash
python scripts/sync_results.py --dir outputs/<...> --experiment <...> --model <...>
hf auth whoami                     # confirm the uploads went to the right account
```

Vast bills for storage on a *stopped* instance, so destroy rather than stop once
results are on the hub.
