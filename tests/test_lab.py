import json
import math
from pathlib import Path

import numpy as np
import pytest

from lenslab.analysis import aggregate, curve, exit_summary, heatmap, load_runs, run_frame
from lenslab.experiments import (BatchBusy, active_batch, batch_records, cpu_threads, create_plan,
                                 gpu_layers, launch_batch, run_worker, start_worker, valid_result)
from lenslab.inventory import SUPPORTED, atomic_json, discover, inspect_model


def measurement(nlls=(2.0, 1.0), count=2):
    return {"schema_version": 2, "model_path": "/models/demo.gguf", "architecture": "llama", "llama_revision": "test",
            "gpu_layers": 0, "requested_tokens": 16, "scored_tokens": count, "native_perplexity": math.e,
            "max_final_layer_nll_error": 0.0, "prompt": "Hi", "answer": "x" * count,
            "layers": [{"layers_kept": i + 1} for i in range(len(nlls))],
            "steps": [{"step": t, "target_piece": "x", "native_nll": 1.0, "target_token": 1,
                       "layer_nll": list(nlls), "layer_argmax": [0, 1]} for t in range(count)]}


def test_weighted_nll_not_average_ppl(tmp_path):
    atomic_json(tmp_path / "a.json", measurement((4.0, 1.0), 1))
    atomic_json(tmp_path / "b.json", measurement((2.0, 1.0), 3))
    runs, errors = load_runs(tmp_path)
    assert not errors
    frame = run_frame(runs)
    pooled = aggregate(frame).sort_values("kept")
    assert pooled.iloc[0]["nll"] == pytest.approx(2.5)
    assert pooled.iloc[0]["perplexity"] == pytest.approx(math.exp(2.5))
    assert pooled.iloc[0]["excess_nll"] == pytest.approx(1.5)
    assert pooled.iloc[0]["tokens"] == 4
    assert len(curve(pooled).data) == 1
    assert exit_summary(pooled).iloc[0]["stable_agreement_depth_pct"] == 100
    assert np.array(heatmap(runs[0]).data[0].z).shape == (2, 1)


def test_corrupt_and_failed_checks_excluded(tmp_path):
    atomic_json(tmp_path / "bad-check.json", {**measurement(), "max_final_layer_nll_error": 0.2})
    (tmp_path / "truncated.json").write_text("{")
    atomic_json(tmp_path / "plan.json", {"tasks": []})
    runs, errors = load_runs(tmp_path)
    assert runs == []
    assert len(errors) == 2


def test_different_runtime_settings_remain_separate(tmp_path):
    atomic_json(tmp_path / "a.json", {**measurement(), "threads": 8})
    atomic_json(tmp_path / "b.json", {**measurement(), "threads": 32})
    runs, _ = load_runs(tmp_path)
    assert run_frame(runs)["series"].nunique() == 2


def test_gpu_suggestion():
    model = {"bytes": 40 * 2**30, "layers": 80}
    assert gpu_layers(model, 0) == 0
    assert 0 < gpu_layers(model, 18) < 80
    assert gpu_layers(model, 50) == 999
    assert cpu_threads() > 0
    assert {"qwen35", "glm5-next", "qwen3moe"} <= SUPPORTED


def test_inspection_and_shard_grouping(tmp_path):
    gguf = pytest.importorskip("gguf")
    first = tmp_path / "Tiny-00001-of-00002.gguf"
    second = tmp_path / "Tiny-00002-of-00002.gguf"
    writer = gguf.GGUFWriter(first, "qwen35")
    writer.add_name("Tiny test model")
    writer.add_block_count(2)
    writer.add_embedding_length(4)
    writer.add_chat_template("test-template")
    writer.add_tensor("token_embd.weight", np.zeros((2, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    missing = inspect_model(first)
    assert not missing["adapter_available"]
    assert missing["missing_shards"] == [str(second)]
    second.write_bytes(b"GGUF")
    item = inspect_model(second)
    assert item["path"] == str(first)
    assert item["adapter_available"] and item["layers"] == 2 and item["width"] == 4
    assert item["bytes"] == first.stat().st_size + second.stat().st_size
    catalog = discover([tmp_path], ollama_root=tmp_path / "no-ollama")
    assert not catalog["errors"]
    assert len(catalog["models"]) == 1


@pytest.fixture
def fake_runner(tmp_path):
    path = tmp_path / "fake-runner"
    script = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv
requests = json.loads(Path(args[args.index('--requests') + 1]).read_text())
counter = Path(__file__).with_suffix('.loads')
counter.write_text(counter.read_text() + '1' if counter.exists() else '1')
for r in requests:
    if r['prompt'] == 'FAIL':
        continue
    result = RESULT
    result['prompt'] = r['prompt']
    p = Path(r['output'])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result))
'''.replace("RESULT", repr(measurement()))
    path.write_text(script)
    path.chmod(0o755)
    return path


def model(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"GGUF")
    return {"id": "test", "name": "Test model", "architecture": "llama", "path": str(weight),
            "shards": [str(weight)], "adapter_available": True, "bytes": 4, "layers": 2}


def test_batch_reuse_model_resume_and_provenance(tmp_path, fake_runner):
    m = model(tmp_path)
    ps = [{"id": "a", "category": "test", "prompt": "one"}, {"id": "b", "prompt": "two"}]
    kwargs = dict(binary=fake_runner, lab=tmp_path / "lab", tokens=16)
    first = create_plan([m], ps, **kwargs)
    assert run_worker(first) == 0
    status = json.loads((first / "status.json").read_text())
    assert status["state"] == "completed"
    assert fake_runner.with_suffix(".loads").read_text() == "1"
    second = create_plan([m], ps, **kwargs)
    assert run_worker(second) == 0
    assert fake_runner.with_suffix(".loads").read_text() == "1"
    paths = list((tmp_path / "lab/runs").glob("*.json"))
    assert len(paths) == 2
    for p in paths:
        r = json.loads(p.read_text())
        assert r["experiment"]["batch_id"] == first.name
        assert r["experiment"]["batch_ids"] == [first.name, second.name]
    assert all(valid_result(p) for p in paths)


def test_failures_and_cancellation_are_visible(tmp_path, fake_runner):
    m = model(tmp_path)
    ps = [{"id": "a", "prompt": "FAIL"}]
    directory = create_plan([m], ps, binary=fake_runner, lab=tmp_path / "lab")
    assert run_worker(directory) == 0
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "completed_with_errors"
    assert next(iter(status["tasks"].values()))["state"] == "failed"
    (directory / "STOP").touch()
    assert run_worker(directory) == 0
    assert json.loads((directory / "status.json").read_text())["state"] == "cancelled"


def test_changed_binary_rejects_stale_plan(tmp_path, fake_runner):
    directory = create_plan([model(tmp_path)], [{"id": "a", "prompt": "one"}],
                            binary=fake_runner, lab=tmp_path / "lab")
    fake_runner.write_text(fake_runner.read_text() + "\n# changed\n")
    assert run_worker(directory) == 1
    assert "changed" in json.loads((directory / "status.json").read_text())["error"]


def test_double_start_and_resume_are_idempotent(tmp_path, fake_runner, monkeypatch):
    import os
    import lenslab.experiments as experiments

    spawned = []

    class Process:
        pid = os.getpid()

        def __init__(self, *args, **kwargs):
            spawned.append(args)

    monkeypatch.setattr(experiments.subprocess, "Popen", Process)
    kwargs = dict(binary=fake_runner, lab=tmp_path / "lab")
    m = model(tmp_path)
    questions = [{"id": "a", "prompt": "one"}]
    first = launch_batch([m], questions, **kwargs)
    assert active_batch(tmp_path / "lab") == first
    with pytest.raises(BatchBusy) as caught:
        launch_batch([m], questions, **kwargs)
    assert caught.value.directory == first
    assert len(batch_records(tmp_path / "lab")) == 1
    assert len(spawned) == 1
    assert start_worker(first) == os.getpid()
    assert len(spawned) == 1


def test_legacy_active_worker_is_recovered(tmp_path, fake_runner):
    import os

    lab = tmp_path / "lab"
    first = create_plan([model(tmp_path)], [{"id": "a", "prompt": "one"}], binary=fake_runner, lab=lab)
    atomic_json(first / "status.json", {"state": "running", "pid": os.getpid()})
    assert active_batch(lab) == first
    with pytest.raises(BatchBusy):
        launch_batch([], [], binary=fake_runner, lab=lab)
    assert len(batch_records(lab)) == 1


def test_simultaneous_clicks_reserve_only_one_batch(tmp_path, fake_runner, monkeypatch):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import lenslab.experiments as experiments

    spawned = []

    class Process:
        pid = os.getpid()

        def __init__(self, *args, **kwargs):
            spawned.append(args)

    monkeypatch.setattr(experiments.subprocess, "Popen", Process)
    m = model(tmp_path)
    barrier = Barrier(2)

    def click():
        barrier.wait()
        try:
            return launch_batch([m], [{"id": "a", "prompt": "one"}], binary=fake_runner, lab=tmp_path / "lab")
        except BatchBusy:
            return "busy"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: click(), range(2)))
    assert results.count("busy") == 1
    assert len(spawned) == 1
    assert len(batch_records(tmp_path / "lab")) == 1
