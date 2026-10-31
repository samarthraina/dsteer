"""The frozen condition table. Every runner reads conditions from here and nowhere else.

A condition is (pair, endpoint side, intervention). Coefficients, layers, token scope and
decoding come from configs/protocol.json, whose SHA-256 is stamped into every output and
is checked against the frozen value at load -- an edited protocol refuses to run rather
than producing outputs under a different identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "configs" / "protocol.json"
PROTOCOL_SHA256 = "3163500b7df3fa5e367f82bc87bd8ea22c27f7b49b3ae87fa3dfdb81d1decb04"

LAYERS = (27, 28, 29, 30, 31)
LAMBDA = {("A", "installation"): 0.05, ("A", "reversal"): -0.05,
          ("B", "installation"): 0.135, ("B", "reversal"): -0.055}
SIDE = {"installation": "it", "reversal": "dpo"}

# Recovered Task 027 files reused verbatim for the harmful panel. Never regenerated.
RECOVERED_HARMFUL = {
    ("A", "it_base"): "M0-A_baseline", ("A", "dpo_base"): "M+-A_baseline",
    ("A", "dsteer_install"): "M0-A_installation", ("A", "dsteer_reverse"): "M+-A_reversal",
    ("B", "it_base"): "M0-B_baseline", ("B", "dpo_base"): "M+-B_baseline",
    ("B", "dsteer_install"): "M0-B_installation", ("B", "dsteer_reverse"): "M+-B_reversal",
}


@dataclass(frozen=True)
class Condition:
    name: str
    side: str                       # "it" | "dpo"
    intervention: str               # none | dsteer_add | refusal_add | refusal_ablate
    direction: Optional[str]        # installation | reversal | None
    multiplier: float = 1.0         # relative to the frozen D-STEER magnitude (refusal_add only)
    mandatory: bool = True
    note: str = ""

    def coefficient(self, pair: str) -> float:
        if self.intervention in ("none", "refusal_ablate"):
            return 0.0 if self.intervention == "none" else 1.0
        return LAMBDA[(pair, self.direction)] * self.multiplier

    def identity(self, pair: str) -> dict:
        return {**asdict(self), "pair": pair, "coefficient": self.coefficient(pair),
                "layers": list(LAYERS) if self.intervention in ("dsteer_add", "refusal_add") else
                          ("all_residual_writes" if self.intervention == "refusal_ablate" else None)}


CONDITIONS = {c.name: c for c in (
    Condition("it_base", "it", "none", None),
    Condition("dpo_base", "dpo", "none", None),
    Condition("dsteer_install", "it", "dsteer_add", "installation"),
    Condition("dsteer_reverse", "dpo", "dsteer_add", "reversal"),
    Condition("refusal_add_install", "it", "refusal_add", "installation",
              note="matched additive: per-layer norm equal to D-STEER's actual perturbation"),
    Condition("refusal_add_reverse", "dpo", "refusal_add", "reversal",
              note="matched additive: per-layer norm equal to D-STEER's actual perturbation"),
    Condition("refusal_ablate_reference", "dpo", "refusal_ablate", "reversal", mandatory=False,
              note="adapted reference directional ablation; NOT additive steering"),
)}
MANDATORY = [c for c in CONDITIONS.values() if c.mandatory]


def tuned_condition(direction: str, multiplier: float) -> Condition:
    """The calibration-selected additive row, labelled distinctly from the matched row."""
    return Condition(f"refusal_add_tuned_{'install' if direction == 'installation' else 'reverse'}",
                     SIDE[direction], "refusal_add", direction, multiplier=multiplier,
                     mandatory=False, note=f"calibration-selected multiplier {multiplier}")


def load_protocol() -> tuple[dict, str]:
    raw = PROTOCOL_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol.json drifted: {sha} != frozen {PROTOCOL_SHA256}. "
                           "Amendments go in a separate dated file; the freeze is never edited.")
    return json.loads(raw), sha


def amendments() -> list[dict]:
    """Dated amendment files, applied in order. The original freeze is never overwritten."""
    d = ROOT / "configs" / "amendments"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
