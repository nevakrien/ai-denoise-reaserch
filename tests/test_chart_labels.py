from lenslab.analysis import model_label
from lenslab.analysis import curve
import pandas as pd
import pytest


def test_model_label_uses_catalog_instead_of_blob_hash():
    run = {"model_path": "/models/sha256-abc", "model_name": "sha256-abc",
           "architecture": "qwen3", "layers": [{}] * 32}
    assert model_label(run, {run["model_path"]: "library/qwen3/8b"}) == "qwen3/8b"
    run["experiment"] = {"model_label": "library/qwen3/8b"}
    assert model_label(run) == "qwen3/8b"


def test_chart_window_preserves_out_of_range_values():
    frame = pd.DataFrame({"series": ["a", "a"], "model": ["Qwen", "Qwen"],
                          "kept": [1, 2], "cut": [1, 0], "depth": [0.5, 1.0],
                          "perplexity": [1e7, 2.0]})
    chart = curve(frame, "perplexity", depth_range=(60, 100), value_range=(0.1, 1000))
    assert tuple(chart.layout.xaxis.range) == (60, 100)
    assert tuple(chart.layout.yaxis.range) == (-1, 3)
    assert list(chart.data[0].y) == [1e7, 2.0]
    linear = curve(frame, "perplexity", value_range=(0, 1000), log_scale=False)
    assert tuple(linear.layout.yaxis.range) == (0, 1000)
    with pytest.raises(ValueError):
        curve(frame, "perplexity", value_range=(0, 1000))


def test_chart_auto_focus_and_locked_axes():
    frame = pd.DataFrame({"series": ["a"] * 3, "model": ["Qwen"] * 3,
                          "kept": [1, 2, 3], "cut": [2, 1, 0], "depth": [0.5, 0.75, 1.0],
                          "perplexity": [1e7, 8.0, 2.0]})
    chart = curve(frame, "perplexity", value_range=(0.1, 10), auto_focus_depth=True)
    assert tuple(chart.layout.xaxis.range) == (75, 100)
    assert chart.layout.xaxis.fixedrange
    assert chart.layout.yaxis.fixedrange
    assert chart.layout.dragmode is False
    assert list(chart.data[0].y) == [1e7, 8.0, 2.0]
