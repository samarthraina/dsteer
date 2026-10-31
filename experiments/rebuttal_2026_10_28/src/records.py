"""Append-only response and score stores with identity binding.

Rules enforced here rather than by convention:
  - an output file is bound to one identity (pair, condition, panel, endpoint, intervention,
    decoding, protocol hash); reopening it under a different identity raises;
  - resume writes only source ids not yet present; a completed record is never overwritten
    or replaced by a preferred rerun;
  - a failed attempt goes to a separate .failures.jsonl and never counts as a sample;
  - a score is keyed by (source_id, response_hash, metric) and refuses to attach to a
    response whose text hash differs from the one it was computed on.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


class IdentityConflict(RuntimeError):
    pass


class DuplicateRecord(RuntimeError):
    pass


class BindingError(RuntimeError):
    pass


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity_hash(identity: dict) -> str:
    return sha256_text(canonical(identity))


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


class _Store:
    key_fields: tuple[str, ...] = ("source_id",)

    def __init__(self, path: str | Path, identity: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.fail_path = self.path.with_suffix(".failures.jsonl")
        self.identity = identity
        self.identity_hash = identity_hash(identity)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if meta.get("identity_hash") != self.identity_hash:
                raise IdentityConflict(
                    f"{self.path}: existing identity {meta.get('identity_hash')} != requested "
                    f"{self.identity_hash}; refusing to mix outputs of different conditions")
        else:
            self.meta_path.write_text(json.dumps(
                {"identity": identity, "identity_hash": self.identity_hash, "created_utc": utc()},
                indent=2, sort_keys=True) + "\n")
        self._keys = {self._key(r) for r in read_jsonl(self.path)}

    def _key(self, rec: dict) -> tuple:
        return tuple(rec[k] for k in self.key_fields)

    def done(self) -> set:
        return set(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def append(self, rec: dict) -> None:
        k = self._key(rec)
        if k in self._keys:
            raise DuplicateRecord(f"{self.path}: {k} already recorded; completed records are immutable")
        rec = {**rec, "identity_hash": self.identity_hash}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        self._keys.add(k)

    def record_failure(self, rec: dict) -> None:
        with self.fail_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**rec, "identity_hash": self.identity_hash, "failed_utc": utc()},
                                sort_keys=True, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        return list(read_jsonl(self.path))

    def finalise_meta(self, **extra) -> None:
        meta = json.loads(self.meta_path.read_text())
        meta.update(extra)
        meta["records"] = len(self._keys)
        meta["failures"] = sum(1 for _ in read_jsonl(self.fail_path))
        meta["file_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None
        meta["finalised_utc"] = utc()
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


class ResponseStore(_Store):
    key_fields = ("source_id",)

    def pending(self, panel: Iterable[dict]) -> list[dict]:
        done = self.done()
        return [r for r in panel if (r["source_id"],) not in done]


class ScoreStore(_Store):
    key_fields = ("source_id", "response_hash", "metric")

    def pending(self, responses: Iterable[dict], metrics: Iterable[str]) -> list[tuple[dict, str]]:
        done = self.done()
        ms = list(metrics)
        return [(r, m) for r in responses for m in ms
                if (r["source_id"], r["response_hash"], m) not in done]

    def append_bound(self, response: dict, score: dict) -> None:
        """Attach a score only if it was computed on exactly this response text."""
        h = sha256_text(response["response"])
        if score.get("response_hash") != h or response.get("response_hash", h) != h:
            raise BindingError(
                f"{response['source_id']}: score bound to {score.get('response_hash')} but the "
                f"response text hashes to {h}")
        self.append(score)


def response_record(*, run_id: str, protocol_hash: str, panel: str, pair: str, condition: dict,
                    endpoint: dict, decoding: dict, prompt_row: dict, result, prompt_tokens: int,
                    started_utc: str, elapsed_s: float, validity: dict) -> dict:
    """The handoff's response schema, in one place so every runner emits the same fields."""
    text = result.text
    return {
        "run_id": run_id, "protocol_hash": protocol_hash, "panel": panel, "pair": pair,
        "condition": condition["name"], "intervention_identity": condition,
        "endpoint_identity": endpoint, "decoding_identity": decoding, "seed": decoding.get("seed"),
        "source_id": prompt_row["source_id"], "prompt_hash": prompt_row["prompt_hash"],
        "prompt": prompt_row["prompt"],
        "response": text, "response_hash": sha256_text(text),
        "finish_reason": result.stop_reason, "stop_token_id": result.stop_token_id,
        "matched_stop_string": result.matched_stop_string,
        "has_post_terminator_continuation": result.has_post_terminator_continuation,
        "input_tokens": prompt_tokens, "output_tokens": result.generated_token_count,
        "validity": validity, "started_utc": started_utc, "finished_utc": utc(),
        "elapsed_s": round(elapsed_s, 4),
    }
