"""Pull a previous run back down from the results dataset.

The counterpart to sync_results.py. Once the hub is the durable copy -- which it has to
be, since an instance that runs out of credit can be reclaimed with its disk -- getting
work back onto a fresh machine has to be as routine as pushing it up. Otherwise the
"results are safe" claim is only half true: safe, but not usable without hand-copying.

    # what is up there
    python scripts/fetch_results.py --list

    # newest run of one experiment/model, into the layout the scripts expect
    python scripts/fetch_results.py --experiment steer_generations \
        --model tulu3_harmfulqa_it --dest outputs/steer/tulu3/harmfulqa/it

    # every steering sweep at once, laid out for score_sweep.py
    python scripts/fetch_results.py --restore-sweeps --dest-root outputs/steer

Runs are stored as runs/<experiment>/<model>/<timestamp>/, so "newest" means the last
timestamp unless --run-id names one.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.utils import setup_logging

DEFAULT_REPO = "samarthraina/dsteer-results"


def list_runs(repo: str) -> Dict[str, Dict[str, List[str]]]:
    """{experiment: {model: [run_id, ...]}} for everything under runs/."""
    api = HfApi()
    tree = api.list_repo_tree(repo, repo_type="dataset", recursive=True, path_in_repo="runs")
    found: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for entry in tree:
        parts = entry.path.split("/")
        if len(parts) >= 4 and parts[0] == "runs":
            found[parts[1]][parts[2]].add(parts[3])
    return {e: {m: sorted(r) for m, r in ms.items()} for e, ms in found.items()}


def fetch_run(repo: str, experiment: str, model: str, dest: Path,
              run_id: Optional[str], log) -> Optional[Path]:
    """Download one run's files into dest (flat, no timestamp directory)."""
    runs = list_runs(repo).get(experiment, {}).get(model)
    if not runs:
        log.error(f"nothing at runs/{experiment}/{model}")
        return None

    chosen = run_id or runs[-1]
    if chosen not in runs:
        log.error(f"run {chosen} not found; have {runs}")
        return None

    prefix = f"runs/{experiment}/{model}/{chosen}/"
    api = HfApi()
    files = [
        f.path for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True,
                                           path_in_repo=prefix)
        if getattr(f, "size", None) is not None
    ]
    if not files:
        log.error(f"run {chosen} is empty")
        return None

    dest.mkdir(parents=True, exist_ok=True)
    for path in files:
        local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/dsteer_fetch")
        target = dest / Path(path).name
        target.write_bytes(Path(local).read_bytes())
    log.info(f"{experiment}/{model}/{chosen}: {len(files)} files -> {dest}")
    return dest


# Steering sweeps are stored flat as <model>_<dataset>_<side>, but the scripts expect
# outputs/steer/<model>/<dataset>/<side>. Model names contain hyphens and sides do not,
# so split from the right.
SWEEP_NAME = re.compile(r"^(?P<model>.+)_(?P<dataset>[^_]+)_(?P<side>it_random|it|dpo)$")


def restore_sweeps(repo: str, dest_root: Path, log) -> int:
    """Pull every steering sweep into the layout score_sweep.py expects."""
    runs = list_runs(repo)
    experiment = "steer_generations" if "steer_generations" in runs else None
    if experiment is None:
        log.error("no steer_generations in the dataset")
        return 0

    n = 0
    for model_key in sorted(runs[experiment]):
        m = SWEEP_NAME.match(model_key)
        if not m:
            log.warning(f"cannot parse {model_key!r}, skipping")
            continue
        dest = dest_root / m["model"] / m["dataset"] / m["side"]
        if fetch_run(repo, experiment, model_key, dest, None, log):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Fetch results back from the hub.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--list", action="store_true", help="Show what is stored and exit.")
    parser.add_argument("--experiment")
    parser.add_argument("--model")
    parser.add_argument("--run-id", help="Defaults to the newest.")
    parser.add_argument("--dest")
    parser.add_argument("--restore-sweeps", action="store_true",
                        help="Fetch every steering sweep into the expected layout.")
    parser.add_argument("--dest-root", default="outputs/steer")
    args = parser.parse_args()

    log = setup_logging()

    if args.list:
        for experiment, models in sorted(list_runs(args.repo).items()):
            print(f"\n{experiment}")
            for model, runs in sorted(models.items()):
                print(f"   {model:<40} {len(runs)} run(s), newest {runs[-1]}")
        return

    if args.restore_sweeps:
        n = restore_sweeps(args.repo, Path(args.dest_root), log)
        log.info(f"restored {n} sweeps into {args.dest_root}")
        return

    if not (args.experiment and args.model and args.dest):
        parser.error("need --experiment, --model and --dest (or --list / --restore-sweeps)")
    fetch_run(args.repo, args.experiment, args.model, Path(args.dest), args.run_id, log)


if __name__ == "__main__":
    main()
