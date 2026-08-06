"""Push the activation tensors to the hub.

`sync_to_hub` excludes activations.pt as regenerable, which it is -- but regenerating
costs a GPU, the model weights, and half an hour, and every geometry result in the paper
is derived from these. On the hub they let a reader redo the rank, cosine, vector
construction and convergence analyses on a laptop.

    python scripts/upload_activations.py --repo samarthraina/dsteer-results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload activation tensors to the hub.")
    parser.add_argument("--repo", default="samarthraina/dsteer-results")
    parser.add_argument("--roots", nargs="+",
                        default=["outputs/layer_profile_response_token",
                                 "outputs/layer_profile_harmfulqa"])
    args = parser.parse_args()

    api = HfApi()
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))

    for root in args.roots:
        for path in sorted(Path(root).glob("*/activations.pt")):
            dest = f"activations/{Path(root).name}/{path.parent.name}/activations.pt"
            if dest in existing:
                print(f"have  {dest}")
                continue
            size = path.stat().st_size / 1e9
            print(f"upload {dest}  ({size:.1f} GB)", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                            repo_id=args.repo, repo_type="dataset")
            print(f"done   {dest}", flush=True)


if __name__ == "__main__":
    main()
