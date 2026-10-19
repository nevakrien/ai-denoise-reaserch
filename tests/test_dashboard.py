from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_all_tabs_without_exception():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Collect data", "Compare models", "Inspect a run", "Batch monitor"]
    assert any(widget.label == "Start experiment batch" and widget.disabled for widget in app.button)


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
    next(w for w in app.button if w.label == "Stop active experiment").click().run()
    assert not app.exception
    assert (directory / "STOP").exists()
    assert app.session_state["monitor_batch"] == directory.name
