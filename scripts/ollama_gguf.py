#!/usr/bin/env python3
"""Resolve a locally cached Ollama model to its GGUF blob; no server needed."""

import argparse
import json
import os
import re
from pathlib import Path


def resolve(root: Path, name: str) -> Path:
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    model, tag = name.rsplit(":", 1)
    parts = model.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", *parts]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    if (len(parts) != 3
            or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in [*parts, tag])):
        raise ValueError("Expected model[:tag], namespace/model[:tag], or registry/namespace/model[:tag]")
    manifest = root / "manifests" / Path(*parts) / tag
    data = json.loads(manifest.read_text())
    layers = [x for x in data["layers"] if x["mediaType"] == "application/vnd.ollama.image.model"]
    if len(layers) != 1:
        raise ValueError("Expected exactly one model blob")
    digest = layers[0]["digest"]
    if not digest.startswith("sha256:") or len(digest) != 71 or any(c not in "0123456789abcdef" for c in digest[7:]):
        raise ValueError("Invalid model blob digest")
    path = root / "blobs" / digest.replace(":", "-")
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Model blob is not GGUF")
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="e.g. llama3.1, qwen3:8b")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama/models")))
    args = parser.parse_args()
    try:
        print(resolve(args.root, args.model))
    except (OSError, ValueError, KeyError) as e:
        parser.exit(1, f"Error: {e}\n")


if __name__ == "__main__":
    main()
