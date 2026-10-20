from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_all_tabs_without_exception():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Collect data", "Compare models", "Inspect a run", "Batch monitor"]
    assert any(widget.label == "Start experiment batch" and widget.disabled for widget in app.button)


def test_nll_transition_statistics_are_visible_and_final_layer_toggle_works(tmp_path, monkeypatch):
    import lenslab
    import lenslab.analysis as analysis

    run = {"_path": str(tmp_path / "demo.json"), "model_path": "/demo.gguf", "model_name": "Demo",
           "architecture": "llama", "llama_revision": "test", "scored_tokens": 1,
           "requested_tokens": 1, "prompt": "Hi", "answer": "x", "native_perplexity": 2.7,
           "max_final_layer_nll_error": 0, "eog_included_in_score": False,
           "layers": [{"teacher_forced_argmax_text": "x"} for _ in range(3)],
           "steps": [{"step": 0, "target_piece": "x", "target_token": 1, "native_nll": 1,
                      "layer_nll": [3, 2, 1], "layer_argmax": [0, 0, 1]}]}
    monkeypatch.setattr(lenslab, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "load_runs", lambda root: ([run], []))
    # The inspect tab offers a raw JSON download.
    (tmp_path / "demo.json").write_text("{}")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert any(s.value == "Does NLL improve at the next layer?" for s in app.subheader)

    def summary():
        return next(d.value for d in app.dataframe if "coin_flip_p_two_sided" in d.value.columns)

    assert summary().iloc[0]["improved_pct"] == 100
    assert summary().iloc[0]["coin_flip_p_two_sided"] == .5
    next(w for w in app.checkbox if w.label == "Exclude final layer from NLL transition statistics").check().run()
    assert not app.exception
    assert summary().iloc[0]["transitions"] == 1
    assert summary().iloc[0]["coin_flip_p_two_sided"] == 1


def test_start_redirects_to_the_exact_run_and_disables_start(tmp_path, monkeypatch):
    import os
    import lenslab.experiments as experiments
    import lenslab.inventory as inventory

    directory = tmp_path / "batches/test-started-run"
    directory.mkdir(parents=True)
    demo = {"id": "test", "name": "Test model", "path": "/test.gguf", "architecture": "llama",
            "gib": 1.0, "bytes": 2**30, "layers": 2, "note": "test", "adapter_available": True}
    prompt = {"id": "a", "prompt": "one"}
    inventory.atomic_json(directory / "plan.json", {"tasks": [{"id": "a", "model": demo, "prompt": prompt}]})
    current = {"directory": None}
    launches = []

    def launch(*args, **kwargs):
        launches.append(1)
        current["directory"] = directory
        inventory.atomic_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "tasks": {"a": {"state": "running"}}})
        return directory

    monkeypatch.setattr(inventory, "load_catalog", lambda: {"models": [demo], "roots": [], "errors": []})
    monkeypatch.setattr(experiments, "active_batch", lambda: current["directory"])
    monkeypatch.setattr(experiments, "launch_batch", launch)
    monkeypatch.setattr(experiments, "batch_records", lambda: [] if not current["directory"] else [
        {"directory": directory, "state": "running", "pid": os.getpid()}])
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=60).run()
    budget = next(w.value for w in app.number_input if w.label.startswith("Approximate GPU"))
    editor_key = f"model_editor_{budget}_1"
    app.session_state[editor_key] = {"edited_rows": {0: {"select": True}}, "added_rows": [], "deleted_rows": []}
    app.run()
    start = next(w for w in app.button if w.label == "Start experiment batch")
    assert not start.disabled
    start.click().run()
    assert not app.exception
    assert len(launches) == 1
    assert app.session_state["view"] == "Batch monitor"
    assert app.session_state["monitor_batch"] == directory.name
    assert app.query_params["batch"] == directory.name
    assert next(w for w in app.button if w.label == "Start experiment batch").disabled
    assert any(w.label == "Stop active experiment" and not w.disabled for w in app.button)
    assert any(f"RUNNING · {directory.name}" in w.value for w in app.warning)
    assert any(f"Worker PID: {os.getpid()}" in w.value for w in app.caption)
    app.run()
    assert app.session_state["monitor_batch"] == directory.name
    assert len(launches) == 1
    # An old bookmarked history selection must not hide access to the live run.
    blocked = tmp_path / "batches/old-blocked-run"
    blocked.mkdir()
    inventory.atomic_json(blocked / "plan.json", {"tasks": []})
    inventory.atomic_json(blocked / "status.json", {"state": "blocked", "tasks": {}})
    monkeypatch.setattr(experiments, "batch_records", lambda: [
        {"directory": directory, "state": "running", "pid": os.getpid()},
        {"directory": blocked, "state": "blocked"}])
    app.run()
    next(w for w in app.selectbox if w.label == "Experiment history").select(blocked.name).run()
    assert any("history selection below is a different experiment" in w.value for w in app.info)
    next(w for w in app.button if w.label == "Open active experiment").click().run()
    assert not app.exception
    assert app.session_state["monitor_batch"] == directory.name
    assert app.query_params["batch"] == directory.name
    next(w for w in app.button if w.label == "Stop active experiment").click().run()
    assert not app.exception
    assert (directory / "STOP").exists()
    assert app.session_state["monitor_batch"] == directory.name
