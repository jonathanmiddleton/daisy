import os
from pathlib import Path
from typing import List

import yaml

import pytest

import training.runner as runner
from tools.helpers import is_mac_os


class FakePopen:
    def __init__(self, cmd: List[str], stdout=None, stderr=None, bufsize=None, universal_newlines=None, env=None):
        # Record for assertions
        self.cmd = cmd
        self._stdout_iter = ["fake line 1\n", "fake line 2\n"]
        self.stdout = self  # act as iterator
        self._returncode = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    # Iterator protocol for stdout
    def __iter__(self):
        return iter(self._stdout_iter)

    def wait(self):
        return self._returncode


@pytest.fixture()
def fake_popen(monkeypatch):
    calls = []

    def _fake_popen(*args, **kwargs):
        p = FakePopen(*args, **kwargs)
        calls.append(p)
        return p

    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen)
    return calls


@pytest.fixture()
def tmp_logfile(monkeypatch, tmp_path):
    # Redirect runner's log file creation into a temp file
    def _setup():
        fp = (tmp_path / "runner.log").open("a", buffering=1, encoding="utf-8")
        return tmp_path / "runner.log", fp

    monkeypatch.setattr(runner, "_setup_log_file", _setup)


def test_split_override_and_cartesian():
    # hyphen is normalized to underscore; bare flag -> true
    assert runner._split_override("--full-windows") == ("full_windows", ["true"])  # bare flag
    assert runner._split_override("foo=1,2,3") == ("foo", ["1", "2", "3"])  # csv values
    assert runner._split_override("--bar=baz") == ("bar", ["baz"])  # prefixed

    # helper to detect commas
    assert runner._has_commas(["1,2"]) is True
    assert runner._has_commas(["1", "2"]) is False


def test_build_torchrun_cmd_basic():
    cmd = runner.build_run_cmd(
        nproc=2,
        config="config/validations/pico-fineweb-edu-single-gpu.yml",
        checkpoint="ckpt.pt",
        extra_long_opts=["--full_windows=true", "--some-flag"],
        singleton_overrides=[("wandb_log", "false")],
        grid_overrides=[("grad_acc_steps", ["2", "3"])],
    )
    # Ensure structure and values are present
    assert cmd[:4] == ["torchrun", "--standalone", "--nproc_per_node=2", "train.py"]
    assert "config/validations/pico-fineweb-edu-single-gpu.yml" in cmd
    assert "--init_checkpoint=ckpt.pt" in cmd
    assert "--full_windows=true" in cmd and "--some-flag" in cmd
    assert "--grid=grad_acc_steps=2,3" in cmd and "--wandb_log=false" in cmd


def test_main_invokes_subprocess_with_env_and_overrides(fake_popen, tmp_logfile, monkeypatch):
    # Provide argv with 2-value override to trigger 2 runs and RUN_ID increment
    argv = [
        "config/validations/pico-fineweb-edu-single-gpu.yml",
        "-n", "1" if is_mac_os() else "2",
        "-p", "ckpt.pt",
        "-s", "7",
        "-r", "10",
        "--full_windows",  # passthrough long opt expanded to true
        "grad_acc_steps=2,3",
        "wandb_log=false",
        "--misc-flag",  # should be passed through as-is
    ]

    # Track environment changes
    monkeypatch.setenv("OMP_NUM_THREADS", "4", prepend=False)

    rc = runner.main(argv)
    assert rc == 0

    # Should have launched a single subprocess; train.py will handle both runs in-process
    assert len(fake_popen) == 1

    # First cmd assertions
    first_cmd = fake_popen[0].cmd
    assert first_cmd[0] == "torchrun" if not is_mac_os() else "python"
    if not is_mac_os():
        assert "--nproc_per_node=2" in first_cmd
        assert first_cmd[3] == "train.py"
    else:
        assert first_cmd[1] == "train.py"
    assert "config/validations/pico-fineweb-edu-single-gpu.yml" in first_cmd
    assert "--init_checkpoint=ckpt.pt" in first_cmd
    # passthrough long opts and overrides present
    assert "--full_windows=true" in first_cmd
    assert "--misc-flag" in first_cmd
    assert "--grid=grad_acc_steps=2,3" in first_cmd
    assert "--wandb_log=false" in first_cmd

    # BEGIN_SHARD and RUN_ID env must be set. RUN_ID is base value; train.py increments internally
    assert os.environ.get("BEGIN_SHARD") == "7"
    assert os.environ.get("RUN_ID") == "10"


def test_main_can_forward_overrides_for_all_config_keys(fake_popen, tmp_logfile):
    # Load a real config YAML and create an override token for every top-level key
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "validations" / "pico-fineweb-edu-single-gpu.yml"
    data = yaml.safe_load(cfg_path.read_text())

    override_tokens = []
    for k, v in data.items():
        # Normalize key name the same way runner/train do (hyphens to underscores)
        key = str(k).replace("-", "_")
        # Choose a simple override value by type
        if isinstance(v, bool):
            val = "false" if v else "true"
        elif isinstance(v, int):
            val = str(v)  # keep as-is
        elif isinstance(v, float):
            val = str(v)
        elif isinstance(v, str):
            val = v  # keep as-is
        else:
            # For lists/dicts or other types, pass an empty YAML of the same container kind
            if isinstance(v, list):
                val = "[]"
            elif isinstance(v, dict):
                val = "{}"
            else:
                # Fallback: stringize
                val = str(v)
        override_tokens.append(f"{key}={val}")

    argv = [str(cfg_path)] + override_tokens

    rc = runner.main(argv)
    assert rc == 0

    # One run should be invoked
    assert len(fake_popen) == 1
    cmd = fake_popen[0].cmd

    # Assert that for every override token there is a corresponding --key=value present in the command
    for tok in override_tokens:
        key, val = tok.split("=", 1)
        expected = f"--{key}={val}"
        assert expected in cmd, f"Missing forwarded override: {expected} in {cmd}"
