"""Token-weighted statistics; never average perplexities arithmetically."""

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_runs(root):
    runs, errors = [], []
    for path in sorted(Path(root).rglob("*.json")):
        if path.name.endswith((".meta.json", ".requests.json")) or path.name in {"plan.json", "status.json"}:
            continue
        try:
            run = json.loads(path.read_text())
            if not isinstance(run, dict) or "layers" not in run or "steps" not in run:
                continue
            if not run["steps"] or run["scored_tokens"] != len(run["steps"]):
                raise ValueError("Invalid scored-token count")
            if not 0 <= run["max_final_layer_nll_error"] <= 0.1:
                raise ValueError("Failed final-layer check")
            n = len(run["layers"])
            if not n or any(len(s["layer_nll"]) != n or len(s["layer_argmax"]) != n for s in run["steps"]):
                raise ValueError("Inconsistent layer arrays")
            if any(not math.isfinite(x) for s in run["steps"] for x in [s["native_nll"], *s["layer_nll"]]):
                raise ValueError("Non-finite NLL")
            run["_path"] = str(path.resolve())
            runs.append(run)
        except (OSError, ValueError, KeyError, TypeError) as e:
            errors.append(f"{path}: {e}")
    return runs, errors


def model_label(run, model_labels=None):
    candidates = [run.get("experiment", {}).get("model_label"),
                  (model_labels or {}).get(run["model_path"]), run.get("model_name")]
    for candidate in candidates:
        if candidate and not candidate.startswith("sha256-"):
            return candidate.removeprefix("library/")
    return f"{run['architecture']} · {len(run['layers'])} layers · {Path(run['model_path']).name[:20]}"


def run_frame(runs, model_labels=None):
    rows = []
    for run in runs:
        exp = run.get("experiment", {})
        label = model_label(run, model_labels)
        model_id = exp.get("model_id", run["model_path"])
        raw = exp.get("settings", {}).get("raw_prompt", run.get("formatted_prompt") == run.get("prompt"))
        config = {"model_id": model_id, "files": exp.get("files", []), "gpu_layers": run.get("gpu_layers", 0),
                  "tokens": run.get("requested_tokens"), "threads": run.get("threads"), "raw_prompt": raw,
                  "binary": exp.get("binary_sha256", run["llama_revision"]), "sampling": run.get("sampling")}
        series = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
        config_label = f"{config['tokens']} tokens / {config['threads']} CPU threads / {config['gpu_layers']} GPU layers / {series[:6]}"
        base = {"run": run["_path"], "series": series, "configuration": config_label,
                "model": label, "architecture": run["architecture"],
                "prompt_id": exp.get("prompt_id", Path(run["_path"]).stem), "category": exp.get("category", "legacy"),
                "batch": exp.get("batch_id", "legacy"), "batches": exp.get("batch_ids", [exp.get("batch_id", "legacy")]),
                "tokens": run["scored_tokens"],
                "requested_tokens": run.get("requested_tokens"), "gpu_layers": run.get("gpu_layers", 0),
                "native_nll": np.mean([s["native_nll"] for s in run["steps"]])}
        for i, layer in enumerate(run["layers"]):
            # Recompute from token data, not rounded summaries.
            nll = np.mean([s["layer_nll"][i] for s in run["steps"]])
            agreement = np.mean([s["layer_argmax"][i] == s["target_token"] for s in run["steps"]])
            rows.append({**base, "kept": i + 1, "cut": len(run["layers"]) - i - 1,
                         "depth": (i + 1) / len(run["layers"]), "nll": nll,
                         "perplexity": np.exp(nll), "excess_nll": nll - base["native_nll"],
                         "agreement": agreement})
    return pd.DataFrame(rows)


def aggregate(frame):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for (series, kept), group in frame.groupby(["series", "kept"], sort=False):
        weights = group["tokens"].to_numpy()
        nll = np.average(group["nll"], weights=weights)
        native = np.average(group["native_nll"], weights=weights)
        rows.append({"series": series, "model": group.iloc[0]["model"], "kept": kept,
                     "configuration": group.iloc[0]["configuration"],
                     "cut": group.iloc[0]["cut"], "depth": group.iloc[0]["depth"],
                     "nll": nll, "perplexity": np.exp(nll), "native_perplexity": np.exp(native),
                     "excess_nll": nll - native, "ppl_ratio": np.exp(nll - native),
                     "agreement": np.average(group["agreement"], weights=weights),
                     "tokens": int(weights.sum()), "runs": len(group),
                     "prompt_nll_min": group["nll"].min(), "prompt_nll_max": group["nll"].max()})
    return pd.DataFrame(rows).sort_values(["series", "kept"])


def exit_summary(aggregated, agreement_threshold=0.8):
    rows = []
    if aggregated.empty:
        return pd.DataFrame()
    for series, group in aggregated.groupby("series", sort=False):
        group = group.sort_values("kept")
        # A transient agreement peak is not a stable early-exit threshold.
        sustained = group["agreement"].iloc[::-1].cummin().iloc[::-1]
        eligible = group[sustained >= agreement_threshold]
        at75 = group.iloc[(group["depth"] - 0.75).abs().argmin()]
        at90 = group.iloc[(group["depth"] - 0.90).abs().argmin()]
        rows.append({"model": group.iloc[0]["model"], "series": series,
                     "configuration": group.iloc[0]["configuration"],
                     "runs": int(group.iloc[0]["runs"]), "tokens": int(group.iloc[0]["tokens"]),
                     "native_PPL": group.iloc[0]["native_perplexity"],
                     "PPL_ratio_at_75pct": at75["ppl_ratio"], "actual_depth_75pct": at75["depth"],
                     "PPL_ratio_at_90pct": at90["ppl_ratio"], "actual_depth_90pct": at90["depth"],
                     "stable_agreement_depth_pct": None if eligible.empty else 100 * eligible.iloc[0]["depth"],
                     "NLL_increase_transitions": int((group["nll"].diff() > 1e-6).sum())})
    return pd.DataFrame(rows)


def curve(frame, metric="excess_nll", *, individual=False, depth_range=(0, 100),
          value_range=None, log_scale=None, auto_focus_depth=False):
    fig = go.Figure()
    group_key = "run" if individual else "series"
    for _, group in frame.groupby(group_key, sort=False):
        group = group.sort_values("depth")
        label = group.iloc[0]["prompt_id"] if individual else group.iloc[0]["model"]
        if not individual:
            # Keep differently configured series distinguishable in the legend.
            variants = frame[frame["model"] == label]["series"].nunique()
            if variants > 1:
                label += f" [{group.iloc[0]['configuration']}]"
        fig.add_trace(go.Scatter(x=group["depth"] * 100, y=group[metric], name=label,
                                 mode="lines+markers", marker_size=3,
                                  customdata=np.column_stack([group["kept"], group["cut"]]),
                                  hovertemplate=f"{label}<br>Depth: %{{x:.2f}}%<br>Layers kept: %{{customdata[0]}}, removed: %{{customdata[1]}}<br>{metric}: %{{y:.10g}}<extra></extra>"))
    labels = {"excess_nll": "Excess NLL vs full model (nats/token)", "perplexity": "Perplexity (log scale)",
              "agreement": "Top-1 agreement with generated target", "ppl_ratio": "Perplexity / full-model perplexity"}
    fig.update_layout(template="plotly_white", xaxis_title="Transformer depth retained (%)",
                      yaxis_title=labels.get(metric, metric), hovermode="closest", height=470,
                       legend=dict(orientation="h", y=-0.25), margin=dict(t=30, b=140),
                       hoverlabel=dict(namelength=-1))
    if log_scale is None:
        log_scale = metric in {"perplexity", "ppl_ratio"}
    if log_scale:
        fig.update_yaxes(type="log")
    if metric == "agreement":
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    visible_depth_range = list(depth_range)
    if value_range is not None:
        lower, upper = value_range
        if lower >= upper or (log_scale and lower <= 0):
            raise ValueError("Chart limits must be increasing and positive for a logarithmic axis")
        if log_scale:
            lower, upper = math.log10(lower), math.log10(upper)
        fig.update_yaxes(range=[lower, upper])
        if auto_focus_depth:
            depth_pct = frame["depth"] * 100
            visible = frame[depth_pct.between(*depth_range) & frame[metric].between(*value_range)]
            if not visible.empty:
                first_depth = float(visible["depth"].min() * 100)
                last_depth = float(visible["depth"].max() * 100)
                if first_depth < last_depth:
                    visible_depth_range = [first_depth, last_depth]
                else:
                    visible_depth_range = [max(depth_range[0], first_depth - 2),
                                           min(depth_range[1], last_depth + 2)]
    fig.update_xaxes(range=visible_depth_range, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(dragmode=False)
    if metric == "perplexity" and not log_scale:
        fig.update_yaxes(title_text="Perplexity")
    return fig


def heatmap(run, excess=True):
    values = np.array([s["layer_nll"] for s in run["steps"]]).T
    if excess:
        values -= np.array([s["native_nll"] for s in run["steps"]])[None, :]
    text = [f"{s['step']}: {s['target_piece']!r}" for s in run["steps"]]
    fig = go.Figure(go.Heatmap(z=values, x=list(range(len(text))), y=list(range(1, values.shape[0] + 1)),
                               customdata=np.tile(text, (values.shape[0], 1)),
                               colorscale="RdBu_r" if excess else "Viridis", zmid=0 if excess else None,
                               colorbar_title="Excess NLL" if excess else "NLL",
                               hovertemplate="%{customdata}<br>Layer %{y}<br>%{z:.3f} nats<extra></extra>"))
    fig.update_layout(template="plotly_white", xaxis_title="Generated target-token position",
                      yaxis_title="Layers retained", height=520, margin=dict(t=20))
    return fig
