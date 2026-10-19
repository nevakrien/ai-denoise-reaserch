"""Persistent, resumable model/prompt jobs. Models run sequentially, never in parallel."""

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lenslab import ROOT
from lenslab.inventory import atomic_json

LAB = ROOT / "results/lab"


class BatchBusy(ValueError):
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        super().__init__(f"An experiment is already starting or running{': ' + self.directory.name if self.directory else ''}")


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        # A zombie no longer owns a live worker even if kill(pid, 0) succeeds.
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def batch_records(lab=LAB):
    records = []
    for directory in sorted((Path(lab) / "batches").glob("*"), reverse=True):
        if not (directory / "plan.json").exists():
            continue
        try:
            status = json.loads((directory / "status.json").read_text())
        except (OSError, ValueError):
            status = {"state": "not_started"}
        records.append({"directory": directory, **status})
    return records


def active_batch(lab=LAB):
    lab = Path(lab).resolve()
    try:
        reservation = json.loads((lab / "active.json").read_text())
        directory = Path(reservation["directory"])
        status = json.loads((directory / "status.json").read_text())
        if status.get("state") in {"starting", "running"} and process_alive(reservation.get("pid")):
            return directory
    except (OSError, ValueError, KeyError):
        pass
    # Recover pre-reservation workers and runs started from the CLI.
    for record in batch_records(lab):
        if record.get("state") in {"starting", "running"} and process_alive(record.get("pid")):
            return record["directory"]
    return None


def _ensure_idle(lab):
    import fcntl

    current = active_batch(lab)
    if current:
        raise BatchBusy(current)
    # Also reject an unknown worker during the small gap before it saves status.
    with (lab / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchBusy()


def _spawn_reserved(directory, lab):
    atomic_json(directory / "status.json", {"state": "starting", "started_at": time.time(), "tasks": {}})
    try:
        with (directory / "worker.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "lenslab", "worker", str(directory)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(lab / "active.json", {"directory": str(directory), "pid": process.pid})
        return process.pid
    except OSError as e:
        atomic_json(directory / "status.json", {"state": "failed", "error": str(e)})
        raise


def launch_batch(models, selected_prompts, **kwargs):
    """Reserve the global slot before creating a plan: double clicks cannot create two batches."""
    import fcntl

    lab = Path(kwargs.get("lab", LAB)).resolve()
    lab.mkdir(parents=True, exist_ok=True)
    with (lab / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _ensure_idle(lab)
        directory = create_plan(models, selected_prompts, **kwargs)
        _spawn_reserved(directory, lab)
        return directory


def cpu_threads():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def prompts():
    return json.loads((ROOT / "config/prompts.json").read_text())


def fingerprint(model):
    return [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size,
             "mtime_ns": Path(p).stat().st_mtime_ns} for p in model.get("shards", [model["path"]])]


def gpu_layers(model, budget_gib):
    if not budget_gib or not model.get("layers"):
        return 0
    if model["bytes"] <= budget_gib * 2**30:
        return 999
    # Conservative first guess only: MoE and attention/FFN sizes need not be uniform.
    return max(0, int(budget_gib * 2**30 / model["bytes"] * model["layers"] * 0.8))


def create_plan(models, selected_prompts, *, binary, tokens=64, threads=None, gpu_budget_gib=0,
                save_hidden=False, raw_prompt=False, timeout=7200, lab=LAB):
    threads = cpu_threads() if threads is None else threads
    binary = Path(binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Not an executable: {binary}")
    if not models or not selected_prompts:
        raise ValueError("Choose at least one model and prompt")
    if not 1 <= tokens <= 4096 or not 1 <= threads <= 256 or timeout <= 0 or gpu_budget_gib < 0:
        raise ValueError("Invalid token/thread/timeout/GPU settings")
    lab = Path(lab).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    directory = lab / "batches" / stamp
    directory.mkdir(parents=True)
    settings = {"tokens": tokens, "threads": threads, "save_hidden": save_hidden,
                "raw_prompt": raw_prompt, "binary": str(binary),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    tasks = []
    for model in models:
        if not model.get("adapter_available"):
            raise ValueError(f"No adapter or incomplete shards: {model['name']}")
        files = fingerprint(model)
        ngl = int(model.get("gpu_layers", gpu_layers(model, gpu_budget_gib)))
        for prompt in selected_prompts:
            if not prompt.get("prompt", "").strip():
                raise ValueError("Empty prompt")
            identity = {"settings": settings, "files": files, "gpu_layers": ngl,
                        "prompt": prompt["prompt"], "measurement_version": 2}
            task_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            tasks.append({"id": task_id, "model": model, "prompt": prompt, "gpu_layers": ngl,
                          "files": files, "output": str(lab / "runs" / f"{task_id}.json")})
    plan = {"id": stamp, "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
            "timeout_per_model": timeout, "tasks": tasks, "lab": str(lab)}
    atomic_json(directory / "plan.json", plan)
    return directory


def start_worker(directory):
    """Resume an existing plan with the same startup reservation as a new batch."""
    import fcntl

    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    lab = Path(plan["lab"])
    with (lab / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = active_batch(lab)
        if current == directory:
            status = json.loads((directory / "status.json").read_text())
            if status.get("pid"):
                return status["pid"]
            return json.loads((lab / "active.json").read_text())["pid"]
        _ensure_idle(lab)
        (directory / "STOP").unlink(missing_ok=True)
        return _spawn_reserved(directory, lab)


def valid_result(path):
    try:
        r = json.loads(Path(path).read_text())
        if not r.get("steps") or not r.get("layers") or not 0 <= r["max_final_layer_nll_error"] <= 0.1:
            return False
        if r.get("scored_tokens") != len(r["steps"]):
            return False
        if "hidden_states" in r:
            h = r["hidden_states"]
            if Path(h["path"]).stat().st_size != math.prod(h["shape"]) * 4:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_worker(directory):
    directory = Path(directory).resolve()
    try:
        return _run_worker(directory)
    except Exception as e:
        status_path = directory / "status.json"
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            status = {}
        status.update(state="failed", error=f"{type(e).__name__}: {e}", updated_at=time.time())
        atomic_json(status_path, status)
        return 1


def _run_worker(directory):
    import fcntl

    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    settings = plan["settings"]
    lab = Path(plan["lab"])
    lab.mkdir(parents=True, exist_ok=True)
    with (lab / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            atomic_json(directory / "status.json", {"state": "blocked", "error": "Another batch is running. Resume after it finishes."})
            return 1
        # Refuse stale plans after a rebuild or a model-file change.
        if hashlib.sha256(Path(settings["binary"]).read_bytes()).hexdigest() != settings["binary_sha256"]:
            atomic_json(directory / "status.json", {"state": "failed", "error": "Runner changed; create a new batch."})
            return 1
        status = {"state": "running", "pid": os.getpid(), "started_at": time.time(),
                  "tasks": {t["id"]: {"state": "pending"} for t in plan["tasks"]}}

        def save():
            status["updated_at"] = time.time()
            atomic_json(directory / "status.json", status)

        def finalize(task):
            path = Path(task["output"])
            if not valid_result(path):
                return False
            data = json.loads(path.read_text())
            old = data.get("experiment", {})
            batches = list(dict.fromkeys([*old.get("batch_ids", []), *([old["batch_id"]] if "batch_id" in old else []), plan["id"]]))
            data["experiment"] = {"task_id": task["id"], "batch_id": old.get("batch_id", plan["id"]), "batch_ids": batches, "model_id": task["model"]["id"],
                                  "model_label": task["model"]["name"], "prompt_id": task["prompt"]["id"],
                                  "category": task["prompt"].get("category", "custom"), "files": task["files"],
                                  "settings": settings,
                                  "binary_sha256": settings["binary_sha256"]}
            atomic_json(path, data)
            return True

        save()
        seen = set()
        for task in plan["tasks"]:
            model_id = task["model"]["id"]
            if model_id in seen:
                continue
            seen.add(model_id)
            group = [t for t in plan["tasks"] if t["model"]["id"] == model_id]
            pending = []
            for t in group:
                if finalize(t):
                    status["tasks"][t["id"]] = {"state": "completed", "cached": True}
                else:
                    pending.append(t)
            if not pending:
                save()
                continue
            if hashlib.sha256(Path(settings["binary"]).read_bytes()).hexdigest() != settings["binary_sha256"]:
                raise ValueError("Runner changed during collection; create a new batch")
            if (directory / "STOP").exists():
                status["state"] = "cancelled"
                break
            if fingerprint(task["model"]) != task["files"]:
                for t in pending:
                    status["tasks"][t["id"]] = {"state": "failed", "error": "Model files changed; create a new batch"}
                save()
                continue
            request_file = directory / f"{model_id}.requests.json"
            atomic_json(request_file, [{"prompt": t["prompt"]["prompt"], "output": t["output"]} for t in pending])
            cmd = [settings["binary"], "--model", task["model"]["path"], "--requests", str(request_file),
                   "--tokens", str(settings["tokens"]), "--threads", str(settings["threads"]),
                   "--gpu-layers", str(task["gpu_layers"])]
            if settings["save_hidden"]:
                cmd.append("--save-hidden")
            if settings["raw_prompt"]:
                cmd.append("--raw-prompt")
            log_path = directory / f"{model_id}.log"
            status["current_model"] = task["model"]["name"]
            status["current_log"] = str(log_path)
            status["command"] = cmd
            for t in pending:
                status["tasks"][t["id"]] = {"state": "running"}
            save()
            started = time.monotonic()
            reason = None
            with log_path.open("a") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        if (directory / "STOP").exists():
                            reason = "cancelled"
                        elif time.monotonic() - started > plan["timeout_per_model"]:
                            reason = "timeout"
                        if reason:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            break
                        for t in pending:
                            if status["tasks"][t["id"]]["state"] == "running" and finalize(t):
                                status["tasks"][t["id"]] = {"state": "completed"}
                        save()
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            for t in pending:
                if finalize(t):
                    status["tasks"][t["id"]] = {"state": "completed"}
                else:
                    status["tasks"][t["id"]] = {"state": "cancelled" if reason == "cancelled" else "failed",
                                                "error": reason or f"Runner exited {process.returncode}; inspect model log"}
            save()
            if reason == "cancelled":
                status["state"] = "cancelled"
                break
        if status["state"] == "running":
            status["state"] = "completed" if all(t["state"] == "completed" for t in status["tasks"].values()) else "completed_with_errors"
        for t in status["tasks"].values():
            if t["state"] == "pending" and status["state"] == "cancelled":
                t["state"] = "cancelled"
        save()
    return 0
