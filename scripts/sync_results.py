"""Push a run directory to the results dataset on the hub.

Scripts that support --sync do this themselves when they finish. This is for the
other cases: a sweep still running that you want backed up before the rental
expires, a run whose analysis you redid locally, or a directory produced before
--sync existed.

    python scripts/sync_results.py --dir outputs/layer_profile/tulu3 \
        --experiment layer_profile --model tulu3

    # keep pushing every 10 minutes while a long sweep writes
    python scripts/sync_results.py --dir outputs/dpo_eval/tulu3 \
        --experiment dpo_eval --model tulu3 --watch 600

Raw activations (~2 GB) are skipped unless --include-weights is passed; they are
regenerable from the config and the checkpoint, and the hub copy is meant for
results other people need to read.

Needs a write token: `hf auth login`, or HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.artifacts import DEFAULT_REPO, sync_to_hub
from steering.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Upload a run directory to the results dataset.")
    parser.add_argument("--dir", required=True, help="Local run directory to upload.")
    parser.add_argument("--experiment", required=True, help="e.g. layer_profile, dpo_eval, steering")
    parser.add_argument("--model", required=True, help="Model pair name, e.g. tulu3")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--run-id",
        help="Defaults to a UTC timestamp. Pass the same id to update an existing run "
             "rather than creating a new one -- which is what --watch does.",
    )
    parser.add_argument("--include-weights", action="store_true", help="Also upload activations.pt.")
    parser.add_argument(
        "--watch", type=int, metavar="SECONDS",
        help="Re-upload on this interval until interrupted. Reuses one run id.",
    )
    args = parser.parse_args()

    log = setup_logging()
    local_dir = Path(args.dir)
    if not local_dir.exists():
        parser.error(f"no such directory: {local_dir}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def push():
        return sync_to_hub(
            local_dir, experiment=args.experiment, model=args.model,
            repo_id=args.repo, run_id=run_id, include_weights=args.include_weights,
        )

    if not args.watch:
        push()
        return

    log.info(f"Watching {local_dir}, pushing every {args.watch}s to run id {run_id}. Ctrl-C to stop.")
    while True:
        try:
            push()
        except Exception as e:  # noqa: BLE001 -- a failed upload should not kill the watcher
            log.warning(f"Upload failed, will retry: {type(e).__name__}: {e}")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            log.info("Stopped.")
            return


if __name__ == "__main__":
    main()
