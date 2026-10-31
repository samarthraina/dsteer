"""Task 027 final held-out HarmfulQA generation driver -- CONFIRMATORY.

scripts/steer_sweep.py is the grid-search runner; its resolve_prompt_partition()
restricts HarmfulQA to ('development','calibration') and states in its own docstring
that "final_evaluation is reserved for the later confirmatory runner, not this grid
search". That runner does not exist at 84b50df4. This file is it, and it lives outside
the repository so the repository is not modified.

The scope guard is not weakened; it is simply not the guard for this runner. Every
integrity check that protects the final panel still executes, inside the repository's
own frozen loader load_harmfulqa_partition():

    validate_manifest_identity   manifest self-hash + lineage
    validate_source_binding      pinned dataset revision, per-prompt hash binding
    sort by permuted_position    frozen order, not this file's choice

and this driver additionally pins the resulting panel to a precomputed ordered
identity digest over (source_id, prompt_hash), so a reordered or substituted panel
aborts before any model is loaded.

Everything scientifically meaningful is imported from the repository:

    resolve_model_source / resolve_all_roles / validate_activation_artifact_for_run
    build_vectors / random_vectors_like / ActivationSteering
    build_chat_prompts / generate_batched / run_one
    load_prompts -> load_harmfulqa_partition
    set_all_seeds(42)

Coefficients are read from the Task 026A paired-v2 selection record. Nothing here
tunes, searches, or reselects anything.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys
from pathlib import Path

REPO = Path("/workspace/dsteer")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch
import steer_sweep as ss
from steering.endpoint_binding import resolve_all_roles
from steering.artifacts import GpuMonitor, write_run_metadata
from steering.generate import STOP_STRINGS, build_chat_prompts
from steering.models import load_model, load_tokenizer
from steering.steer import build_vectors, random_vectors_like
from steering.utils import set_all_seeds, setup_logging

AMENDMENT = Path("/workspace/dsteer-data/protocol-amendments/"
                 "2026-08-18-task027-final-evaluation-runner-v1.json")
SELECTION = Path("/workspace/dsteer-data/outputs-026a/analysis-paired-v2/selection_record_v2.json")

PARTITION = "final_evaluation"
PANEL_N = 300
HARMFULQA_MANIFEST_HASH = "0e1d0a8177d59eba601141a74da19a189527821dab573e203a3261cff54eec70"
PANEL_ORDERED_DIGEST = "7851299f7e15743e47c0e4604697a0770b8e9607a9dd4420a0f31a3529a53953"
CONTROL_SEEDS = (11, 22, 33, 44, 55)
GATE_FRACTION = 0.05

SOURCE_ROOTS = {
    "pair_a_sft": Path("/workspace/dsteer-data/sources/pair_a_sft_repo/SFT_merged"),
    "pair_a_dpo": Path("/workspace/dsteer-data/sources/pair_a_dpo_repo/DPO_merged"),
    "pair_b_sft": Path("/workspace/dsteer-data/sources/pair_b_sft_repo"),
}


def panel_digest(records):
    return hashlib.sha256(
        "\n".join(f"{r['source_id']}:{r['prompt_hash']}" for r in records).encode()
    ).hexdigest()


def gen_gate_from_file(path):
    from steering.validity import check
    n = n_invalid = n_maxlen = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if not check(r.get("response", "")).ok:
            n_invalid += 1
        if r.get("stop_reason") == "max_new_tokens":
            n_maxlen += 1
    cap = GATE_FRACTION * n
    return {"n": n, "n_invalid": n_invalid, "n_maxlen": n_maxlen,
            "invalid_rate": n_invalid / n, "maxlen_rate": n_maxlen / n,
            "gate_cap": cap, "pass": (n_invalid <= cap and n_maxlen <= cap)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=["A", "B"])
    ap.add_argument("--side", required=True, choices=["it", "dpo"])
    ap.add_argument("--magnitude", required=True, type=float)
    ap.add_argument("--eval-config", default="/workspace/dsteer-data/runtime-configs-027/steer_sweep_final.yaml")
    ap.add_argument("--endpoint-manifest",
                    default="/workspace/dsteer-data/endpoints/primary-v1/endpoint_manifest_candidate_v1.json")
    ap.add_argument("--endpoint-bundle-root", default="/workspace/dsteer-data/endpoints/primary-v1")
    ap.add_argument("--random-seed", type=int, default=None, choices=list(CONTROL_SEEDS))
    ap.add_argument("--role", default=None, choices=["M0-A", "M+-A", "M--A", "M0-B", "M+-B"])
    ap.add_argument("--out-root", required=True, help="one writer per directory")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--hourly-rate", type=float, default=1.164)
    args = ap.parse_args()

    amendment = json.loads(AMENDMENT.read_text())
    selection = json.loads(SELECTION.read_text())["frozen_selections_full_precision"]

    tag = "installation" if args.side == "it" else "reversal"
    mag = float(args.magnitude)

    # Coefficients are read, never chosen. A steered arm must equal the frozen lambda*.
    if mag != 0.0:
        frozen = selection[f"{args.pair}_{tag}"]["lambda_star_magnitude"]
        assert abs(mag - frozen) < 1e-12, \
            f"{args.pair}_{tag}: magnitude {mag!r} is not the frozen lambda* {frozen!r}"
    if args.role:
        assert mag == 0.0, "--role is for UNSTEERED arms only"
    if args.random_seed is not None:
        assert mag != 0.0, "a random control needs the same nonzero coefficient as its real arm"

    endpoint_source = (
        [f"pair_a_sft={SOURCE_ROOTS['pair_a_sft']}", f"pair_a_dpo={SOURCE_ROOTS['pair_a_dpo']}"]
        if args.pair == "A" else [f"pair_b_sft={SOURCE_ROOTS['pair_b_sft']}"]
    )

    # --- provenance binding before any side effect ---
    if args.role:
        rs = resolve_all_roles(args.endpoint_manifest, SOURCE_ROOTS, args.endpoint_bundle_root)
        ep = rs.roles[args.role]
        from steering.config import ModelConfig
        model_cfg = ModelConfig(name=f"endpoint-{args.role}-{rs.candidate_manifest_hash[:12]}",
                                base_model=f"endpoint:{args.role}",
                                it_model=str(ep.local_path), dpo_model=str(ep.local_path),
                                architecture=ep.validation["model_type"],
                                num_layers=ep.validation["num_hidden_layers"])
        endpoint_meta = {"role": args.role, "candidate_manifest_hash": rs.candidate_manifest_hash,
                         "source_manifest_hash": rs.source_manifest_hash,
                         "endpoint": ep.as_metadata()}
    else:
        model_cfg, endpoint_meta = ss.resolve_model_source(
            model_config=None, endpoint_manifest=args.endpoint_manifest,
            endpoint_bundle_root=args.endpoint_bundle_root, pair=args.pair,
            endpoint_source=endpoint_source)

    cfg = ss.SteerSweepConfig.from_yaml(args.eval_config)
    assert cfg.prompt_source == "harmfulqa" and cfg.prompt_partition == PARTITION, \
        f"eval-config must declare harmfulqa/{PARTITION}, got {cfg.prompt_source}/{cfg.prompt_partition}"
    layers = ss.resolve_effective_layers(argparse.Namespace(layers=None), cfg, model_cfg)
    assert layers == [27, 28, 29, 30, 31], f"unexpected layers {layers}"
    loading_policy = ss._model_loading_policy(True)
    art = None if args.role else ss.validate_activation_artifact_for_run(cfg, model_cfg, endpoint_meta)

    # The repository's own frozen loader. resolve_prompt_partition -- the grid-search
    # scope guard -- is deliberately not the gate for the confirmatory runner.
    records = ss.load_prompts(cfg.prompt_source, cfg.n_prompts, 42, PARTITION)
    panel = ss.build_prompt_panel_metadata(cfg.prompt_source, records)
    digest = panel_digest(records)
    assert panel["record_count"] == PANEL_N, panel
    assert panel["partition"] == PARTITION, panel
    assert panel["manifest_hash"] == HARMFULQA_MANIFEST_HASH, panel
    assert digest == PANEL_ORDERED_DIGEST, f"final-panel identity digest drifted: {digest}"
    assert len({r["source_id"] for r in records}) == PANEL_N, "duplicate source_id in final panel"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sign = 1.0 if (args.role or args.side == "it") else -1.0
    coeff = sign * abs(mag)
    name = f"lambda_{coeff:+.3f}"

    write_run_metadata(
        out_root,
        config={"model": vars(model_cfg) if hasattr(model_cfg, "__dict__") else str(model_cfg),
                "eval": {k: getattr(cfg, k) for k in
                         ("prompt_source", "prompt_partition", "layers_last_k", "vector_method",
                          "vector_normalise", "positions", "preserve_norm", "max_new_tokens",
                          "max_input_length", "activations_dir", "output_dir")},
                "condition": args.condition, "magnitude": mag, "lambda_signed": coeff,
                "side": args.side, "pair": args.pair, "direction": tag},
        extra={
            "task": "027", "label": "final held-out evaluation, CONFIRMATORY",
            "protocol_profile": "final_confirmatory_v1",
            "protocol_profile_note": (
                "Final held-out HarmfulQA panel. Coefficients, layers, normalisation, decoding, "
                "validity logic and control seeds are frozen from Task 026A/026B and are read, "
                "never selected, here. Generated by the out-of-repo confirmatory runner; the "
                "repository is unmodified."),
            "amendment_file": str(AMENDMENT), "amendment_hash": amendment["amendment_hash"],
            "selection_record": str(SELECTION),
            "selection_record_sha256": hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
            "frozen_lambda_star": selection.get(f"{args.pair}_{tag}", {}).get("lambda_star_signed"),
            "endpoint_backed": True, "endpoint": endpoint_meta,
            "model_loading_policy": loading_policy, "resolved_layers": layers,
            "stop_strings": list(STOP_STRINGS), "random_control_seed": args.random_seed,
            "control_seeds_frozen": list(CONTROL_SEEDS),
            "harmfulqa_partition": PARTITION, "harmfulqa_manifest_hash": panel["manifest_hash"],
            "harmfulqa_record_count": panel["record_count"], "prompt_panel": panel,
            "panel_ordered_identity_digest": digest,
            "activation_artifact": ({"path": str(art.acts_path), "sha256": art.sha256,
                                     "size_bytes": art.size_bytes,
                                     "run_identity_hash": art.sidecar.get("run_identity_hash")}
                                    if art is not None else None),
            "explicit_role": args.role, "prompt_source": cfg.prompt_source,
            "condition": args.condition,
        },
        argv=list(sys.argv),
    )

    set_all_seeds(42)
    log = setup_logging(out_root / "gen_final.log")
    log.info(f"Task 027 condition={args.condition} pair={args.pair} side={args.side} "
             f"lambda={coeff:+.3f} seed={args.random_seed} role={args.role}")
    log.info(f"panel={PARTITION} n={panel['record_count']} digest={digest}")
    log.info(f"amendment_hash={amendment['amendment_hash']}")

    vectors = None
    if art is not None:
        vectors = build_vectors(art.acts_path, method=cfg.vector_method, layers=layers,
                                normalise=cfg.vector_normalise, skip_first=0)
    if args.random_seed is not None:
        assert vectors is not None, "random control requires a real direction to match"
        real_norms = {L: float(v.norm()) for L, v in vectors.items()}
        vectors = random_vectors_like(vectors, seed=args.random_seed)
        rand_norms = {L: float(v.norm()) for L, v in vectors.items()}
        assert all(abs(real_norms[L] - rand_norms[L]) < 1e-4 for L in vectors), "norm match failed"
        log.info(f"RANDOM CONTROL seed={args.random_seed}: norm-matched isotropic Gaussian per "
                 f"layer via repo-frozen random_vectors_like; norms {rand_norms}")

    which = model_cfg.it_model if (args.role or args.side == "it") else model_cfg.dpo_model
    tokenizer = load_tokenizer(model_cfg.tokenizer_id or which, subfolder="", **loading_policy)
    model = load_model(which, subfolder="", **loading_policy)
    chats = build_chat_prompts(tokenizer, [r["prompt"] for r in records])
    batch_size = cfg.batch_size or ss.suggest_batch_size(model)
    log.info(f"batch size {batch_size}")

    path = out_root / f"{name}.jsonl"
    with GpuMonitor(out_root, hourly_rate=args.hourly_rate) as gpu:
        n = ss.run_one(model, tokenizer, records, chats, coeff, vectors, cfg,
                       path, f"{args.condition} {name}", batch_size, mode="add")
    g = gen_gate_from_file(path)
    log.info(f"{name}: +{n} records -> {path} | invalid {g['n_invalid']}/{g['n']} "
             f"maxlen {g['n_maxlen']}/{g['n']} usability_gate_pass={g['pass']}")
    log.info(f"GPU usage: {json.dumps(gpu.summary())}")
    (out_root / "generation_gates.json").write_text(json.dumps({args.condition: g}, indent=2))
    print(f"GATE {args.condition}: invalid={g['n_invalid']}/{g['n']} "
          f"maxlen={g['n_maxlen']}/{g['n']} pass={g['pass']}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    # A failed usability gate is REPORTED, never acted on: no arm is dropped or retried.
    print("GENERATION_DONE", json.dumps({"condition": args.condition, "file": str(path),
                                         "n": g["n"], "gate_pass": g["pass"]}), flush=True)


if __name__ == "__main__":
    main()
