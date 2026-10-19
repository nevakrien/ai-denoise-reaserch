import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ollama_gguf", Path(__file__).parents[1] / "scripts/ollama_gguf.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resolve(tmp_path):
    digest = "sha256:" + "a" * 64
    manifest = tmp_path / "manifests/registry.ollama.ai/library/demo/latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}]}))
    blob = tmp_path / "blobs" / digest.replace(":", "-")
    blob.parent.mkdir()
    blob.write_bytes(b"GGUFtest")
    assert module.resolve(tmp_path, "demo") == blob
    assert module.resolve(tmp_path, "library/demo:latest") == blob
    blob.write_bytes(b"oops")
    with pytest.raises(ValueError, match="not GGUF"):
        module.resolve(tmp_path, "demo")


@pytest.mark.parametrize("name", ["../demo", "demo:../bad", "/tmp/demo", "a/b/c/d"])
def test_reject_path_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        module.resolve(tmp_path, name)
