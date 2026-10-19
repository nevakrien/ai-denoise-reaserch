"""Discover local weights and inspect GGUF metadata without loading model weights."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from lenslab import ROOT

CATALOG = ROOT / "config/models.local.json"
SUPPORTED = {"llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe", "qwen35", "qwen35moe",
             "glm4moe", "glm-dsa", "glm5-next"}
SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def inspect_model(path, alias=None):
    from gguf import GGUFReader

    path = Path(path).expanduser().resolve()
    match = SPLIT.match(path.name)
    if match and int(match[2]) != 1:
        path = path.with_name(f"{match[1]}-00001-of-{match[3]}.gguf")
    reader = GGUFReader(path)

    def field(name, default=None):
        item = reader.get_field(name)
        return default if item is None else item.contents()

    arch = field("general.architecture", "unknown")
    name = field("general.name", path.stem)
    match = SPLIT.match(path.name)
    shards = ([path.with_name(f"{match[1]}-{i:05d}-of-{match[3]}.gguf")
               for i in range(1, int(match[3]) + 1)] if match else [path])
    missing = [str(p) for p in shards if not p.is_file()]
    size = sum(p.stat().st_size for p in shards if p.is_file())
    return {"id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
            "name": alias or name, "gguf_name": name, "path": str(path), "architecture": arch,
            "bytes": size, "gib": round(size / 2**30, 2), "shards": [str(p) for p in shards],
            "missing_shards": missing, "layers": field(f"{arch}.block_count"),
            "width": field(f"{arch}.embedding_length"), "file_type": field("general.file_type"),
            "has_chat_template": bool(field("tokenizer.chat_template")),
            "adapter_available": arch in SUPPORTED and not missing,
            "note": "Missing shards" if missing else ("Per-run final-layer validation required" if arch in SUPPORTED else "No adapter")}


def discover(roots=(), ollama_root=None):
    candidates = {}
    errors = []
    ollama = Path(ollama_root or os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama/models"))
    if (ollama / "manifests").is_dir():
        for manifest in sorted((ollama / "manifests").rglob("*")):
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text())
                alias = str(manifest.relative_to(ollama / "manifests/registry.ollama.ai"))
                for layer in data.get("layers", []):
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        path = ollama / "blobs" / layer["digest"].replace(":", "-")
                        candidates.setdefault(str(path.resolve()), alias)
            except (OSError, ValueError, KeyError) as e:
                errors.append(f"{manifest}: {e}")
    for root in roots:
        root = Path(root).expanduser()
        paths = [root] if root.is_file() else sorted(root.rglob("*.gguf"))
        for path in paths:
            if any(part in {".git", ".deps", "build"} for part in path.parts):
                continue
            match = SPLIT.match(path.name)
            if match and int(match[2]) != 1:
                continue
            if path.name.startswith(("ggml-vocab-", "mmproj", "incomplete-")):
                continue
            candidates.setdefault(str(path.resolve()), None)
    models = []
    for path, alias in candidates.items():
        try:
            models.append(inspect_model(path, alias))
        except Exception as e:
            errors.append(f"{path}: {e}")
    return {"models": sorted(models, key=lambda m: m["bytes"]), "errors": errors,
            "roots": [str(r) for r in roots]}


def load_catalog():
    if CATALOG.exists():
        return json.loads(CATALOG.read_text())
    return {"models": [], "errors": [], "roots": []}
