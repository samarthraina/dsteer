"""Guards on `start_judge_server.py`'s command construction (Task 010).

`build_command`/`frozen_command` are pure -- no subprocess is ever started here, so the
frozen judge-server command (protocol Section 10) can be checked field-by-field without
launching vLLM. `verify_vllm_version` is exercised with an injected version getter, so
none of this needs vLLM actually installed in the CPU test environment.

Run with:
    pytest tests/test_start_judge_server.py -v

CPU-only, offline: no subprocess is ever started.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import start_judge_server  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steering.judge_identity import FROZEN_JUDGE_IDENTITY, served_model_alias  # noqa: E402


# build_command


def test_build_command_contains_the_standard_flags():
    cmd = start_judge_server.build_command(
        model="org/model", port=8001, gpu_util=0.8, dtype="bfloat16", api_key="EMPTY",
        max_model_len=4096, prefix_caching=True,
    )
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "org/model"
    assert "--dtype" in cmd and cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert "--max-model-len" in cmd and cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8001"
    assert "--api-key" in cmd and cmd[cmd.index("--api-key") + 1] == "EMPTY"
    assert "--enable-prefix-caching" in cmd


def test_build_command_omits_prefix_caching_when_disabled():
    cmd = start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=4096, prefix_caching=False,
    )
    assert "--enable-prefix-caching" not in cmd


def test_build_command_includes_revision_served_name_runner_and_language_model_only_when_given():
    cmd = start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=8192, prefix_caching=True,
        revision="a" * 40, served_model_name="org/model@" + "a" * 40,
        runner="generate", language_model_only=True,
    )
    assert cmd[cmd.index("--revision") + 1] == "a" * 40
    assert cmd[cmd.index("--served-model-name") + 1] == "org/model@" + "a" * 40
    assert cmd[cmd.index("--runner") + 1] == "generate"
    assert "--language-model-only" in cmd
    assert "--task" not in cmd


def test_build_command_omits_runner_and_language_model_only_when_not_given():
    cmd = start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=8192, prefix_caching=True,
    )
    assert "--runner" not in cmd
    assert "--language-model-only" not in cmd
    assert "--task" not in cmd


def test_build_command_never_adds_a_quantization_flag_when_none():
    cmd = start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=8192, prefix_caching=True, quantization=None,
    )
    assert "--quantization" not in cmd


def test_build_command_never_adds_speculative_decoding_or_mtp_flags():
    cmd = start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=8192, prefix_caching=True,
        revision="a" * 40, served_model_name="alias", runner="generate", language_model_only=True,
    )
    joined = " ".join(cmd)
    assert "speculative" not in joined.lower()
    assert "mtp" not in joined.lower()
    assert "num-speculative-tokens" not in joined


def test_build_command_never_starts_a_subprocess(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("build_command must never launch a subprocess")

    monkeypatch.setattr(start_judge_server.subprocess, "Popen", boom)
    start_judge_server.build_command(
        model="org/model", port=8000, gpu_util=0.9, dtype="bfloat16", api_key="EMPTY",
        max_model_len=8192, prefix_caching=True,
    )  # must not raise / must not touch subprocess.Popen


# verify_vllm_version


def test_verify_vllm_version_accepts_the_matching_version():
    installed = start_judge_server.verify_vllm_version(expected="0.26.0", version_getter=lambda name: "0.26.0")
    assert installed == "0.26.0"


def test_verify_vllm_version_rejects_a_mismatched_version():
    with pytest.raises(start_judge_server.VllmVersionError):
        start_judge_server.verify_vllm_version(expected="0.26.0", version_getter=lambda name: "0.25.0")


def test_verify_vllm_version_rejects_a_missing_package():
    def boom(name):
        raise ModuleNotFoundError(name)

    with pytest.raises(start_judge_server.VllmVersionError):
        start_judge_server.verify_vllm_version(expected="0.26.0", version_getter=boom)


# frozen_command


def test_frozen_command_refuses_to_build_on_a_version_mismatch():
    with pytest.raises(start_judge_server.VllmVersionError):
        start_judge_server.frozen_command(
            port=8000, gpu_util=0.9, api_key="EMPTY", version_getter=lambda name: "0.25.0",
        )


def test_frozen_command_never_starts_a_subprocess_on_a_version_mismatch(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("frozen_command must never launch a subprocess")

    monkeypatch.setattr(start_judge_server.subprocess, "Popen", boom)
    with pytest.raises(start_judge_server.VllmVersionError):
        start_judge_server.frozen_command(
            port=8000, gpu_util=0.9, api_key="EMPTY", version_getter=lambda name: "0.25.0",
        )


def test_frozen_command_contains_every_frozen_field():
    cmd = start_judge_server.frozen_command(
        port=8000, gpu_util=0.9, api_key="EMPTY", version_getter=lambda name: "0.26.0",
    )
    f = FROZEN_JUDGE_IDENTITY
    repository, revision = f["judge"]["repository"], f["judge"]["revision"]
    expected_alias = served_model_alias(repository, revision)

    assert cmd[cmd.index("--model") + 1] == repository
    assert cmd[cmd.index("--revision") + 1] == revision
    assert cmd[cmd.index("--served-model-name") + 1] == expected_alias
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert cmd[cmd.index("--max-model-len") + 1] == str(f["max_model_len"])
    assert cmd[cmd.index("--runner") + 1] == "generate"
    assert "--language-model-only" in cmd
    assert "--task" not in cmd
    assert "--quantization" not in cmd  # frozen quantization is None
    assert "--enable-prefix-caching" in cmd


def test_frozen_command_never_launches_a_real_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("frozen_command must never launch a subprocess")

    monkeypatch.setattr(start_judge_server.subprocess, "Popen", boom)
    start_judge_server.frozen_command(
        port=8000, gpu_util=0.9, api_key="EMPTY", version_getter=lambda name: "0.26.0",
    )  # must not raise / must not touch subprocess.Popen


# main(): --frozen wiring


def test_main_with_frozen_flag_builds_the_frozen_command_before_launching(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["start_judge_server.py", "--frozen"])
    monkeypatch.setattr(
        start_judge_server, "verify_vllm_version", lambda expected=None, version_getter=None: "0.26.0",
    )

    captured = {}

    class _FakeProc:
        stdout = SimpleNamespace(readline=lambda: "")  # iter(readline, "") ends immediately

        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(start_judge_server.subprocess, "Popen", fake_popen)
    start_judge_server.main()

    f = FROZEN_JUDGE_IDENTITY
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == f["judge"]["repository"]
    assert captured["cmd"][captured["cmd"].index("--runner") + 1] == "generate"
    assert "--language-model-only" in captured["cmd"]
    assert "--task" not in captured["cmd"]


def test_main_with_frozen_flag_never_launches_on_a_version_mismatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["start_judge_server.py", "--frozen"])

    def fake_verify_vllm_version(expected=None, version_getter=None):
        raise start_judge_server.VllmVersionError("mismatch")

    monkeypatch.setattr(start_judge_server, "verify_vllm_version", fake_verify_vllm_version)

    def boom(*a, **k):
        raise AssertionError("must not launch a subprocess on a version mismatch")

    monkeypatch.setattr(start_judge_server.subprocess, "Popen", boom)
    with pytest.raises(start_judge_server.VllmVersionError):
        start_judge_server.main()
