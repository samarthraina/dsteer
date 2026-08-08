"""Train the steering vector itself through the DPO loss, with the model frozen.

Every vector here so far is *read off* a pair of checkpoints: the mean displacement
between them. That is a description of what preference optimisation did, and there is no
reason it should be the best direction for producing the same behaviour. The obvious
comparison is a vector optimised for the objective directly.

So: freeze the model, add one vector per steered layer to the residual stream, and train
only those vectors on preference pairs under the DPO loss. About 20k parameters against a
frozen 7B. If a learned vector reaches further than the mean difference, the ceiling we
measured belongs to that particular direction rather than to steering; if it does not,
the limit is structural and a much stronger claim.

Two things make this cheap. The reference policy is the same frozen model with the vector
switched off, so its log-probabilities never change and are computed once up front rather
than every step. And the graph only begins where the vector enters, so the layers below
the first hook cost nothing to back through.

    python scripts/train_steering_vector.py --model-config configs/llama3_oh_local.yaml \
        --side it --init outputs/layer_profile_response_token/llama3-oh/activations.pt \
        --output-dir outputs/learned_vector --steps 2000

Writes vectors.pt as {layer: tensor}, which `--vectors` on the sweep and the ceiling
search reads directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from steering.activations import decoder_layers
from steering.artifacts import write_run_metadata
from steering.config import ModelConfig
from steering.data import load_hh_rlhf_test
from steering.models import load_model, load_tokenizer
from steering.steer import build_vectors, steered_layers
from steering.utils import set_all_seeds, setup_logging


class TrainableSteering:
    """Adds a learnable vector to the residual stream at each chosen layer.

    Separate from `ActivationSteering`, which holds fixed tensors and is used at
    inference. Here the vectors are parameters and the hooks must stay in the graph, so
    the two paths are kept apart rather than one being bent to serve both.
    """

    def __init__(self, model, layers: Sequence[int], init: Dict[int, torch.Tensor] | None = None):
        self.model = model
        self.layers = list(layers)
        blocks = decoder_layers(model)
        device = next(model.parameters()).device
        hidden = model.config.hidden_size

        self.params: Dict[int, torch.Tensor] = {}
        for layer in self.layers:
            if init is not None and layer in init:
                v = init[layer].to(device=device, dtype=torch.float32).clone()
            else:
                v = torch.zeros(hidden, device=device, dtype=torch.float32)
            self.params[layer] = v.requires_grad_(True)

        self._blocks = blocks
        self._handles: List = []
        self.enabled = False

    def _hook(self, layer: int):
        def fn(_module, _inputs, output):
            if not self.enabled:
                return output
            v = self.params[layer]
            if isinstance(output, tuple):
                return (output[0] + v.to(output[0].dtype),) + output[1:]
            return output + v.to(output.dtype)
        return fn

    def __enter__(self):
        for layer in self.layers:
            self._handles.append(self._blocks[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def vectors(self) -> Dict[int, torch.Tensor]:
        return {l: v.detach().float().cpu() for l, v in self.params.items()}


def sequence_logprob(model, ids: torch.Tensor, mask: torch.Tensor,
                     prompt_lens: torch.Tensor) -> torch.Tensor:
    """Summed log-probability of the response tokens in each row.

    Prompt tokens are excluded: the same prompt appears under both the chosen and the
    rejected continuation, so including it adds a constant to both sides of the
    preference comparison and shrinks the difference the loss is built from.
    """
    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    logits = out.logits[:, :-1]
    targets = ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(targets.shape[1], device=ids.device).unsqueeze(0)
    response = (positions >= (prompt_lens.unsqueeze(1) - 1)) & mask[:, 1:].bool()
    return (token_logp * response).sum(dim=-1)


def encode(tokenizer, prompt: str, response: str, max_length: int):
    p = tokenizer(prompt, add_special_tokens=True).input_ids
    full = tokenizer(prompt + response, add_special_tokens=True).input_ids[:max_length]
    return full, min(len(p), len(full))


def collate(batch, pad_id: int, device):
    width = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    plens = torch.zeros(len(batch), dtype=torch.long)
    for i, (seq, plen) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        plens[i] = plen
    return ids.to(device), mask.to(device), plens.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train a steering vector under the DPO loss.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--side", choices=["it", "dpo"], default="it",
                        help="Checkpoint to steer. 'it' asks what a learned vector can "
                             "install that the mean difference could not.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init", default=None, metavar="ACTIVATIONS.PT",
                        help="Initialise from the mean displacement in this file. Zeros "
                             "otherwise, which starts the policy equal to the reference.")
    parser.add_argument("--layers-last-k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Far larger than a weight learning rate: this optimises a few "
                             "thousand parameters, not billions.")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-pairs", type=int, default=8000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    model_cfg = ModelConfig.from_yaml(args.model_config)
    out = Path(args.output_dir) / model_cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "train_steering_vector.log")
    write_run_metadata(out, config={"model": asdict(model_cfg), **vars(args)})

    which = model_cfg.it_model if args.side == "it" else model_cfg.dpo_model
    sub = (model_cfg.it_subfolder if args.side == "it" else model_cfg.dpo_subfolder) or ""
    tokenizer = load_tokenizer(model_cfg.tokenizer_id or which,
                               subfolder=model_cfg.tokenizer_subfolder or sub)
    model = load_model(which, subfolder=sub)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device

    records = load_hh_rlhf_test(n=args.n_pairs, seed=args.seed)
    pairs = []
    for r in records:
        if not r.get("chosen") or not r.get("rejected"):
            continue
        prompt = r["prompt"] if isinstance(r["prompt"], str) else str(r["prompt"])
        pairs.append((encode(tokenizer, prompt, r["chosen"], args.max_length),
                      encode(tokenizer, prompt, r["rejected"], args.max_length)))
    log.info(f"{len(pairs)} preference pairs")

    layers = steered_layers(model_cfg.num_layers, args.layers_last_k)
    init = None
    if args.init:
        init = build_vectors(Path(args.init), method="mean", layers=layers)
        log.info(f"Initialised from {args.init}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with TrainableSteering(model, layers, init) as steering:
        params = list(steering.params.values())
        n_param = sum(p.numel() for p in params)
        log.info(f"{n_param} trainable parameters across layers {layers[0]}-{layers[-1]}")
        opt = torch.optim.AdamW(params, lr=args.lr)

        # The reference policy is this same frozen model with the vector switched off, so
        # its log-probabilities are fixed. Caching them halves the forward passes.
        ref_cache: Dict[int, tuple] = {}
        history = []
        order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed))

        step = 0
        bar = tqdm(total=args.steps, desc="training", unit="step")
        while step < args.steps:
            for start in range(0, len(pairs), args.batch_size):
                if step >= args.steps:
                    break
                idx = order[start:start + args.batch_size].tolist()
                if not idx:
                    continue
                chosen = collate([pairs[i][0] for i in idx], pad_id, device)
                rejected = collate([pairs[i][1] for i in idx], pad_id, device)

                key = start
                if key not in ref_cache:
                    steering.enabled = False
                    with torch.no_grad():
                        ref_cache[key] = (sequence_logprob(model, *chosen),
                                          sequence_logprob(model, *rejected))
                ref_c, ref_r = ref_cache[key]

                steering.enabled = True
                pol_c = sequence_logprob(model, *chosen)
                pol_r = sequence_logprob(model, *rejected)

                logits = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
                loss = -F.logsigmoid(logits).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                step += 1
                bar.update(1)
                if step % args.log_every == 0 or step == 1:
                    row = {
                        "step": step,
                        "loss": loss.item(),
                        "accuracy": (logits > 0).float().mean().item(),
                        "margin": logits.mean().item(),
                        "norm": torch.sqrt(sum((p.detach() ** 2).sum() for p in params)).item(),
                    }
                    history.append(row)
                    log.info(f"step {step:5d}  loss {row['loss']:.4f}  "
                             f"acc {row['accuracy']:.3f}  margin {row['margin']:+.3f}  "
                             f"|v| {row['norm']:.3f}")
        bar.close()
        vectors = steering.vectors()

    torch.save(vectors, out / "vectors.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    log.info(f"Saved {len(vectors)} vectors to {out}/vectors.pt")

    if init is not None:
        log.info("\nlayer   cos(learned, mean)   |learned| / |mean|")
        for layer in layers:
            a, b = vectors[layer], init[layer]
            log.info(f"{layer:5d}   {(a / a.norm() @ (b / b.norm())).item():+.4f}"
                     f"              {(a.norm() / b.norm()).item():.3f}")


if __name__ == "__main__":
    main()
