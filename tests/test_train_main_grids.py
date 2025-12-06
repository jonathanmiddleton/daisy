import os

import pytest


def test_parse_grid_and_inprocess_runs(monkeypatch):
    # Import train after monkeypatch targets resolved
    import train as train_mod

    created_runtimes = []
    runs = []

    class DummyRuntime:
        def __init__(self, args_for_group, dynamic: bool):
            created_runtimes.append({
                "dynamic": dynamic,
                "args": args_for_group,
            })

        def reset_model_to_initial(self):
            pass

        def destroy(self):
            pass

    class DummySession:
        def __init__(self, runtime, args, run_id: int):
            runs.append({"run_id": run_id, "args": args})

        def run(self):
            # no-op
            return None

    # Patch train entrypoints to dummies
    monkeypatch.setattr(train_mod, "CompiledRuntime", DummyRuntime)
    monkeypatch.setattr(train_mod, "TrainingSession", DummySession)

    # Ensure deterministic run ids
    monkeypatch.setenv("RUN_ID", "5")

    # Two values for lr_scale => 2 in-process runs handled by single subprocess
    argv = [
        "../config/test/test_tiny_model.yml",
        "--grid", "lr_scale=0.5,1.0",
        # plus a singleton override forwarded through positional list
        "wandb_log=false",
    ]

    rc = train_mod.main(argv)
    assert rc == 0

    # One runtime created for the compile group
    assert len(created_runtimes) == 1
    # Two runs executed with incrementing run ids starting at RUN_ID
    assert [r["run_id"] for r in runs] == [5, 6]

    # Group max_seq_len elevation: ensure each args passed to session has max_seq_len >= any val shard seq len
    for r in runs:
        a = r["args"]
        max_val_len = 0
        for v in getattr(a, "val_shards", []) or []:
            max_val_len = max(max_val_len, int(v.get("sequence_length")))
        if max_val_len > 0:
            assert int(getattr(a, "max_seq_len", a.training_sequence_length)) >= max_val_len


def test__parse_grid_item(monkeypatch):
    import train as train_mod

    parse = train_mod._parse_grid_item
    k, vals = parse("lr_scale=0.1,1.0,2.0")
    assert k == "lr_scale" and vals == ["0.1", "1.0", "2.0"]
    k, vals = parse("--grid=foo=bar")
    assert k == "foo" and vals == ["bar"]
    with pytest.raises(ValueError):
        parse("--grid=novalue")
