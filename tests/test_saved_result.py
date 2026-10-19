"""Validate a real run with GGUF_LENS_RESULT=results/run.json python -m pytest."""

import json
from array import array
import math
import os
from pathlib import Path

import pytest


@pytest.fixture
def result():
    path = os.environ.get("GGUF_LENS_RESULT")
    if not path:
        pytest.skip("Set GGUF_LENS_RESULT to validate an actual model run")
    return json.loads(Path(path).read_text())


def test_saved_metrics(result):
    steps = result["steps"]
    layers = result["layers"]
    assert len(steps) == result["scored_tokens"] > 0
    assert result["max_final_layer_nll_error"] <= 0.1
    expected_native = math.exp(sum(s["native_nll"] for s in steps) / len(steps))
    assert result["native_perplexity"] == pytest.approx(expected_native)
    for i, layer in enumerate(layers):
        assert layer["layers_kept"] == i + 1
        assert layer["layers_cut"] == len(layers) - i - 1
        mean_nll = sum(s["layer_nll"][i] for s in steps) / len(steps)
        assert math.isfinite(mean_nll) and mean_nll >= 0
        assert layer["mean_nll"] == pytest.approx(mean_nll)
        assert layer["perplexity"] == pytest.approx(math.exp(mean_nll))
        agreement = sum(s["layer_argmax"][i] == s["target_token"] for s in steps) / len(steps)
        assert layer["top1_agreement"] == pytest.approx(agreement)
    for i, step in enumerate(steps):
        assert step["context_position"] == len(result["prompt_token_ids"]) - 1 + i
        assert len(step["layer_nll"]) == len(step["layer_argmax"]) == len(layers)
        assert abs(step["layer_nll"][-1] - step["native_nll"]) <= 0.1
    assert result["answer"] == "".join(s["target_piece"] for s in steps if not s["is_eog"])


def test_saved_hidden(result):
    if "hidden_states" not in result:
        pytest.skip("Run with --save-hidden")
    meta = result["hidden_states"]
    assert meta["shape"][:2] == [result["scored_tokens"], len(result["layers"])]
    assert Path(meta["path"]).stat().st_size == math.prod(meta["shape"]) * 4
    values = array("f")
    with Path(meta["path"]).open("rb") as stream:
        values.fromfile(stream, math.prod(meta["shape"]))
    assert all(math.isfinite(x) for x in values)
    assert any(x != 0 for x in values)
