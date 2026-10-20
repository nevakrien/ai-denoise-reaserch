"""Run with: .venv/bin/python -m streamlit run dashboard.py"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lenslab import ROOT
from lenslab.analysis import aggregate, curve, exit_summary, heatmap, load_runs, model_label, run_frame
from lenslab.experiments import (LAB, BatchBusy, active_batch, batch_records, cpu_threads,
                                 gpu_layers, launch_batch, process_alive, prompts, start_worker)
from lenslab.inventory import CATALOG, atomic_json, discover, load_catalog
from lenslab.trends import nll_result_style, nll_transition_tables, trend_figure, trend_summary

st.set_page_config(page_title="GGUF Lens Lab", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1500px; padding-top: 2rem;}
[data-testid="stMetric"] {background: #f0f4fa; border-radius: 12px; padding: 14px;}
</style>""", unsafe_allow_html=True)
st.title("GGUF Lens Lab")
st.caption("Explore where a model's predictions emerge · local GGUFs · layer-wise logit lens")
st.info("All depths score the same full-model-generated tokens. This is self-generated-answer perplexity, "
        "not held-out perplexity or independently generated early-exit text. All blocks still run.")


@st.cache_data(ttl=5)
def cached_runs(root):
    return load_runs(root)


def tail(path, limit=16000):
    try:
        with Path(path).open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode(errors="replace")
    except OSError:
        return "No log yet."


def focus_batch(directory):
    st.session_state["pending_focus"] = Path(directory).name


def launch_clicked(models, questions, settings):
    try:
        directory = launch_batch(models, questions, **settings)
        focus_batch(directory)
        st.session_state["flash"] = ("success", "Experiment started. You are viewing its live monitor.")
    except BatchBusy as e:
        if e.directory:
            focus_batch(e.directory)
        st.session_state["flash"] = ("warning", "An experiment is already running. No second batch was created.")
    except (OSError, ValueError) as e:
        st.session_state["flash"] = ("error", str(e))


def stop_clicked(directory):
    (Path(directory) / "STOP").touch()
    focus_batch(directory)
    st.session_state["flash"] = ("warning", "Stop requested. Completed measurements are kept.")


def chart_view(metric, key):
    with st.expander("Chart viewing range", expanded=True):
        depth_range = st.slider("Depth retained (%)", 0, 100, (0, 100), key=f"{key}_depth")
        auto_focus_depth = st.checkbox("Automatically focus depth on values within the Y range",
                                       value=True, key=f"{key}_auto_depth")
        positive_metric = metric in {"perplexity", "ppl_ratio"}
        log_scale = st.checkbox("Logarithmic Y axis", value=positive_metric, key=f"{key}_log_{metric}")
        mode = st.selectbox("Y-axis range", ["Focus on useful values", "Full range", "Custom"],
                            index=0 if positive_metric else 1, key=f"{key}_range_{metric}")
        focus_ranges = {"perplexity": (0.1, 10.0), "ppl_ratio": (0.1, 100.0),
                        "excess_nll": (0.0, 10.0), "agreement": (0.0, 1.0)}
        value_range = None
        if mode != "Full range":
            lower, upper = focus_ranges[metric]
            if mode == "Custom":
                minimum, maximum = st.columns(2)
                lower = minimum.number_input("Y minimum", value=lower, key=f"{key}_min_{metric}")
                upper = maximum.number_input("Y maximum", value=upper, key=f"{key}_max_{metric}")
            if lower >= upper or (log_scale and lower <= 0):
                st.warning("Use increasing limits, with a positive minimum for a logarithmic axis. Showing full Y range instead.")
            else:
                value_range = (lower, upper)
                st.caption(f"Showing Y = {lower:g} to {upper:g}. Values outside this window are off-screen, not clamped or removed.")
        st.caption("Use these controls to change the view; mouse zoom and pan are disabled. Automatic focus narrows depth to observations inside the selected Y range. Tables, statistics, and CSVs always keep all values.")
    return dict(depth_range=depth_range, value_range=value_range, log_scale=log_scale,
                auto_focus_depth=auto_focus_depth)


current_batch = active_batch()
if "pending_focus" in st.session_state:
    batch_name = st.session_state.pop("pending_focus")
    st.session_state["view"] = "Batch monitor"
    st.session_state["monitor_batch"] = batch_name
    st.query_params["batch"] = batch_name
if "view" not in st.session_state and "view" not in st.query_params:
    st.session_state["view"] = "Batch monitor" if current_batch or st.query_params.get("batch") else "Collect data"
if "monitor_batch" not in st.session_state:
    st.session_state["monitor_batch"] = st.query_params.get("batch", current_batch.name if current_batch else "")
if "flash" in st.session_state:
    kind, message = st.session_state.pop("flash")
    getattr(st, kind)(message)
@st.fragment(run_every="3s")
def active_experiment_panel():
    directory = active_batch()
    if not directory:
        st.caption("No experiment worker is running.")
        return
    status = json.loads((directory / "status.json").read_text())
    tasks = status.get("tasks", {})
    complete = sum(task["state"] == "completed" for task in tasks.values())
    failed = sum(task["state"] == "failed" for task in tasks.values())
    with st.container(border=True):
        st.warning(f"RUNNING · {directory.name}")
        st.write(f"Current model: **{status.get('current_model', 'Starting worker / loading model')}**")
        st.caption(f"Worker PID: {status.get('pid', 'starting')} · "
                   f"{complete}/{len(tasks)} completed · {failed} failed · updates every 3 seconds")
        st.caption(f"Run directory: {directory}")
        if (directory / "STOP").exists():
            st.warning("Stop requested. Waiting for the runner to exit; completed results are kept.")
        open_col, stop_col = st.columns(2)
        if open_col.button("Open active experiment", key="open_active"):
            focus_batch(directory)
            st.rerun(scope="app")
        stop_col.button("Stop active experiment", on_click=stop_clicked, args=(directory,), key="stop_active")


active_experiment_panel()


with st.sidebar:
    st.header("Workspace")
    result_root = st.text_input("Results directory", str(ROOT / "results"))
    if st.button("Refresh results"):
        cached_runs.clear()
    st.divider()
    st.markdown("**Interpretation**")
    st.caption("Compare normalized depth across architectures. Excess NLL removes each run's native "
               "baseline but does not remove tokenizer or self-generation bias. Do not interpret this as a model-quality ranking.")
    st.caption("Aggregates pool token NLLs, then exponentiate. Each model/configuration is kept separate. "
               "Repeated prompts are not independent statistical replications.")

collect, compare, inspect_tab, jobs = st.tabs(["Collect data", "Compare models", "Inspect a run", "Batch monitor"],
                                           key="view", bind="query-params", on_change="rerun")

with collect:
    catalog = load_catalog()
    with st.expander("Discover GGUFs / add model directories", expanded=not catalog["models"]):
        roots = st.text_area("Directories or individual GGUF files, one per line",
                             "\n".join(catalog.get("roots", [])),
                             help="Ollama's local cache is always included. No model downloads. Split GGUFs are grouped.")
        if st.button("Scan local models"):
            with st.spinner("Reading GGUF metadata (not model weights)..."):
                catalog = discover([p.strip() for p in roots.splitlines() if p.strip()])
                atomic_json(CATALOG, catalog)
            st.rerun()
        if catalog.get("errors"):
            st.warning("Some files could not be inspected:")
            st.code("\n".join(catalog["errors"]))
    left, right = st.columns([1, 1])
    with left:
        binary_default = ROOT / ("build-cuda/gguf-lens" if (ROOT / "build-cuda/gguf-lens").exists() else "build/gguf-lens")
        binary = st.text_input("Runner executable", str(binary_default))
        budget = st.number_input("Approximate GPU weight budget (GiB; 0 = CPU)", 0.0, 1000.0,
                                 18.0 if "cuda" in binary else 0.0, step=1.0)
        st.caption("GPU layer suggestions are estimates; override per model below. Leave VRAM for KV cache and compute. "
                   "Use a CUDA-built executable for GPU runs. Very large models also need ample RAM.")
    with right:
        tokens = st.number_input("Generated tokens per question", 1, 4096, 96, step=16)
        threads = st.number_input("CPU threads", 1, max(256, cpu_threads()), cpu_threads())
        timeout = st.number_input("Timeout per model, all prompts (minutes)", 1, 1440, 120)
    st.subheader("1 · Models")
    model_rows = [{"select": False, "id": m["id"], "name": m["name"], "architecture": m["architecture"],
                   "GiB": m["gib"], "layers": m["layers"], "adapter": m["adapter_available"],
                   "gpu_layers": gpu_layers(m, budget), "note": m["note"]} for m in catalog["models"]]
    if model_rows:
        edited = st.data_editor(pd.DataFrame(model_rows), hide_index=True, width="stretch",
                                disabled=["id", "name", "architecture", "GiB", "layers", "adapter", "note"],
                                column_config={"select": st.column_config.CheckboxColumn("Run"),
                                               "gpu_layers": st.column_config.NumberColumn("GPU layers", min_value=0, max_value=999)},
                                key=f"model_editor_{budget}_{len(model_rows)}")
        chosen = []
        for row in edited.to_dict("records"):
            if row["select"]:
                model = next(m for m in catalog["models"] if m["id"] == row["id"]).copy()
                model["gpu_layers"] = int(row["gpu_layers"])
                chosen.append(model)
    else:
        st.warning("Scan your model directories to begin.")
        chosen = []
    st.subheader("2 · Prompts")
    suite = prompts()
    options = {p["id"]: p for p in suite}
    selected_ids = st.multiselect("Built-in suite", list(options), default=[p["id"] for p in suite],
                                 format_func=lambda x: f"{options[x]['category']} · {x}")
    with st.expander("Preview / edit selected questions", expanded=False):
        selected_prompts = []
        for key in selected_ids:
            p = options[key].copy()
            p["prompt"] = st.text_area(f"{p['category']} · {key}", p["prompt"], key=f"prompt_{key}")
            selected_prompts.append(p)
    custom = st.text_area("Additional question (optional)", placeholder="Ask your own question...")
    if custom.strip():
        selected_prompts.append({"id": "custom", "category": "custom", "prompt": custom})
    hidden = st.checkbox("Save raw hidden states (can consume substantial disk space)")
    raw = st.checkbox("Use raw prompts (skip chat formatting)", help="For base models or manually formatted prompts only.")
    heavy = any(m["gib"] > 30 for m in chosen)
    consent = st.checkbox("I have enough RAM/VRAM for the selected models above 30 GiB") if heavy else True
    st.caption(f"{len(chosen)} models × {len(selected_prompts)} questions = {len(chosen) * len(selected_prompts)} runs. "
               "Each model loads once. Completed matching runs are reused; failures remain visible and can be resumed.")
    st.button("Start experiment batch", type="primary",
              disabled=bool(current_batch) or not chosen or not selected_prompts or not consent,
              on_click=launch_clicked,
              args=(chosen, selected_prompts, dict(binary=binary, tokens=int(tokens), threads=int(threads),
                    gpu_budget_gib=budget, save_hidden=hidden, raw_prompt=raw, timeout=int(timeout) * 60)))
    if current_batch:
        st.caption("Already running — use Open active experiment above to monitor or stop it. No need to click Start again.")

runs, read_errors = cached_runs(result_root)
with compare:
    if not runs:
        st.info("No completed measurements yet. Start a batch in Collect data.")
    else:
        model_labels = {m["path"]: m["name"] for m in load_catalog()["models"]}
        frame = run_frame(runs, model_labels)
        batches = sorted({b for bs in frame["batches"] for b in bs}, reverse=True)
        batch_filter = st.multiselect("Batches to compare", batches, default=[batches[0]])
        selected = frame[frame["batches"].apply(lambda bs: bool(set(bs) & set(batch_filter)))]
        categories = st.multiselect("Prompt categories", sorted(selected["category"].unique()),
                                    default=sorted(selected["category"].unique()))
        selected = selected[selected["category"].isin(categories)]
        model_filter = st.multiselect("Models", list(selected["model"].unique()), default=list(selected["model"].unique()))
        selected = selected[selected["model"].isin(model_filter)]
        # Do not count a cached task twice when it is referenced by multiple batches.
        selected = selected.drop_duplicates(["run", "kept"])
        agg = aggregate(selected)
        if not agg.empty:
            a, b, c, d = st.columns(4)
            a.metric("Model/config series", agg["series"].nunique())
            b.metric("Completed runs", selected["run"].nunique())
            c.metric("Scored tokens", int(selected.drop_duplicates("run")["tokens"].sum()))
            d.metric("Prompt categories", selected["category"].nunique())
            metric = st.selectbox("Curve", ["excess_nll", "perplexity", "agreement", "ppl_ratio"])
            view = chart_view(metric, "comparison")
            fig = curve(agg, metric, **view)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
            st.download_button("Download comparison as interactive HTML", fig.to_html(include_plotlyjs=True),
                               "depth-comparison.html", "text/html")
            st.download_button("Download aggregate CSV", agg.to_csv(index=False), "depth-statistics.csv", "text/csv")
            st.subheader("Does NLL improve at the next layer?")
            st.caption("ΔNLL = NLL(i+1) − NLL(i) on each token-weighted model/configuration curve. "
                       "Negative = improved; positive = worsened. Percentages include all transitions; "
                       "The test compares improved (ΔNLL < 0) against not improved (ΔNLL ≥ 0), including exact zeros. "
                       "Batch, category, and model filters apply; chart viewing limits do not.")
            exclude_final_changes = st.checkbox("Exclude final layer from NLL transition statistics", value=False)
            change_summary, change_details = nll_transition_tables(agg, exclude_final_changes)
            display_changes = change_summary.drop(columns=["series", "coin_flip_p_one_sided", "coin_flip_p_not_improved", "non_tied"])
            styled_changes = display_changes.style.apply(
                lambda row: [nll_result_style(row["signed_p_display"])] * len(row), axis=1)
            st.dataframe(styled_changes, hide_index=True, width="stretch",
                         column_config={
                             "improved_pct": st.column_config.NumberColumn("Improved / down (%)", format="%.2f"),
                             "not_improved_pct": st.column_config.NumberColumn("Not improved / up or zero (%)", format="%.2f"),
                             "worsened_pct": st.column_config.NumberColumn("Worsened / up (%)", format="%.2f"),
                             "tied_pct": st.column_config.NumberColumn("Tied (%)", format="%.2f"),
                             "coin_flip_p_two_sided": st.column_config.NumberColumn("Coin-flip p (two-sided)", format="%.6g"),
                             "signed_p_display": st.column_config.NumberColumn("Signed p (+ improved / − not improved)", format="%+.6g"),
                             "mean_delta_nll": st.column_config.NumberColumn("Mean ΔNLL", format="%.6g")})
            st.caption("H₀: each transition independently improves with probability 0.5. "
                       "H₁: improvement probability ≠ 0.5. Exact two-sided p = min(1, 2 × smaller directional tail). "
                       "No transitions → p is undefined. P-values are unadjusted across model comparisons.")
            st.caption("Signed p is a display convention: + means more improved, − means more not-improved transitions; "
                       "its absolute value is the ordinary two-sided p-value (probabilities themselves are never negative). "
                       "Balanced counts have no direction. Bright red marks any negative signed p (more not improved); "
                       "green marks non-negative signed p < 0.05 (rejecting the null in favor of improvement). "
                       "Yellow marks inconclusive results or no transitions; "
                       "more data may help but does not guarantee significance. Red takes priority for any not-improved majority. "
                       "The test measures directional bias, "
                       "not serial dependence of signs.")
            st.download_button("Download NLL transition summary CSV", change_summary.to_csv(index=False),
                               "nll-transition-summary.csv", "text/csv")
            st.download_button("Download every NLL transition CSV", change_details.to_csv(index=False),
                               "nll-transitions.csv", "text/csv")
            with st.expander("Individual adjacent-layer NLL changes"):
                st.dataframe(change_details.drop(columns="series", errors="ignore"), hide_index=True, width="stretch")
            st.subheader("Exact values by model and depth")
            st.caption("Select a model/configuration to read every layer. CSV exports retain full numeric precision.")
            series_options = list(agg["series"].unique())
            series_labels = {s: f"{g.iloc[0]['model']} · {g.iloc[0]['configuration']}"
                             for s, g in agg.groupby("series", sort=False)}
            value_series = st.selectbox("Model/configuration for numerical values", series_options,
                                        format_func=series_labels.get)
            values = agg[agg["series"] == value_series].sort_values("kept").copy()
            values["depth_pct"] = values["depth"] * 100
            values = values[["model", "kept", "cut", "depth_pct", "perplexity", "native_perplexity",
                             "ppl_ratio", "excess_nll", "agreement", "runs", "tokens"]]
            st.dataframe(values, hide_index=True, width="stretch", column_config={
                name: st.column_config.NumberColumn(format="%.8g")
                for name in ["depth_pct", "perplexity", "native_perplexity", "ppl_ratio", "excess_nll", "agreement"]})
            st.download_button("Download this model's values as CSV", values.to_csv(index=False),
                               "model-depth-values.csv", "text/csv")
            with st.expander("Depth trend fits: linear, monotonic, or a step?"):
                st.caption("Fits use every selected layer, independent of chart viewing limits. "
                           "NLL is log(perplexity); raw perplexity can be dominated by early-layer extremes. "
                           "Spearman measures rank trend (−1 for decreasing, +1 for increasing). "
                           "Isotonic regression allows curves, plateaus, and steps, but no wrong-way motion.")
                trend_metric = st.selectbox("Fit metric", ["nll", "perplexity", "agreement"])
                omit_final = st.checkbox("Exclude final layer from fit", value=False)
                trend_data = agg[agg["series"] == value_series]
                if len(trend_data) - int(omit_final) >= 3:
                    st.plotly_chart(trend_figure(trend_data, trend_metric, omit_final), width="stretch")
                else:
                    st.info("At least three layers are needed for a fit.")
                trends = trend_summary(agg)
                st.dataframe(trends.drop(columns="series", errors="ignore"), hide_index=True, width="stretch")
                st.download_button("Download trend fits CSV", trends.to_csv(index=False),
                                   "depth-trends.csv", "text/csv")
                st.warning("All R² values are in-sample descriptions, not evidence of causality. "
                           "Isotonic has more flexibility, and the step location is selected on these same data. "
                           "The full-model greedy target makes final-layer agreement essentially guaranteed. "
                           "A fixed external corpus and prompt-level validation are needed to test generalization.")
            st.subheader("Early-exit summary")
            threshold = st.slider("Sustained top-1 agreement threshold", 0.5, 1.0, 0.8, step=0.05)
            st.caption("Earliest depth meeting the threshold at every remaining depth. This is a descriptive statistic "
                       "of these generated samples, not a learned stopping rule. 75%/90% values use the nearest actual block.")
            st.dataframe(exit_summary(agg, threshold).drop(columns="series"), hide_index=True, width="stretch")
            st.subheader("Per-model detail")
            for series, data in agg.groupby("series", sort=False):
                label = data.iloc[0]["model"]
                with st.expander(f"{label} · {int(data.iloc[0]['runs'])} runs · {int(data.iloc[0]['tokens'])} scored tokens", expanded=agg["series"].nunique() == 1):
                    st.caption(data.iloc[0]["configuration"])
                    sub = selected[selected["series"] == series]
                    detail_view = chart_view("perplexity", f"detail_{series}")
                    col1, col2 = st.columns(2)
                    col1.plotly_chart(curve(sub, "perplexity", individual=True, **detail_view), width="stretch",
                                     key=f"ppl_{series}", config={"displayModeBar": False, "scrollZoom": False})
                    col2.plotly_chart(curve(data, "agreement"), width="stretch", key=f"agree_{series}")
                    increases = int((data.sort_values("kept")["nll"].diff() > 1e-6).sum())
                    st.caption(f"Aggregate NLL increases at {increases} adjacent-depth transitions; monotonicity is not enforced. "
                               "Colored prompt lines show variation, not confidence intervals.")
                    st.dataframe(data[["kept", "cut", "perplexity", "ppl_ratio", "excess_nll", "agreement"]], hide_index=True)
        else:
            st.info("Select at least one completed batch, category, and model.")
    if read_errors:
        with st.expander("Unreadable or invalid results (excluded from charts)"):
            st.code("\n".join(read_errors))

with inspect_tab:
    if runs:
        by_path = {r["_path"]: r for r in reversed(runs)}

        def run_label(path):
            r = by_path[path]
            exp = r.get("experiment", {})
            labels = {m["path"]: m["name"] for m in load_catalog()["models"]}
            return f"{model_label(r, labels)} · {exp.get('prompt_id', Path(path).stem)} · {r['scored_tokens']} tokens"

        selected_path = st.selectbox("Completed run", list(by_path), format_func=run_label)
        run = by_path[selected_path]
        st.markdown("**Question**")
        st.text(run["prompt"])
        st.markdown("**Full-model answer**")
        st.text(run["answer"])
        st.caption(f"Final-layer max NLL error: {run['max_final_layer_nll_error']:.3g} nat · "
                   f"Native PPL: {run['native_perplexity']:.4g} · EOG reached: {run['eog_included_in_score']} · "
                   f"Residual streams: {run.get('residual_streams', 1)}")
        excess = st.checkbox("Subtract native NLL in heatmap", value=True)
        st.plotly_chart(heatmap(run, excess), width="stretch")
        depth = st.slider("Inspect teacher-forced predictions at depth", 1, len(run["layers"]), len(run["layers"]))
        st.text(run["layers"][depth - 1]["teacher_forced_argmax_text"])
        st.caption("This text concatenates guesses under the full model's prefixes; it is NOT a free-running early-exit answer.")
        rows = [{"position": s["step"], "target_id": s["target_token"], "target_piece": repr(s["target_piece"]),
                 "prediction_id": s["layer_argmax"][depth - 1], "NLL": s["layer_nll"][depth - 1],
                 "native_NLL": s["native_nll"]} for s in run["steps"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Provenance and formatted prompt"):
            st.json({k: v for k, v in run.items() if k not in {"steps", "layers", "answer"}})
        st.download_button("Download raw measurement JSON", Path(selected_path).read_bytes(), Path(selected_path).name, "application/json")
    else:
        st.info("No completed runs to inspect.")


@st.fragment(run_every="3s")
def batch_monitor():
    records = batch_records()
    if not records:
        st.info("No batches yet.")
        return
    by_name = {r["directory"].name: r for r in records}
    running = active_batch()
    if running and st.session_state.get("monitor_batch") != running.name:
        active_status = json.loads((running / "status.json").read_text())
        st.info(f"The running experiment is {running.name} · "
                f"{active_status.get('current_model', 'Starting worker')}. "
                "The history selection below is a different experiment.")
        if st.button("Monitor the running experiment", key="monitor_running"):
            focus_batch(running)
            st.rerun(scope="app")
    if st.session_state.get("monitor_batch") not in by_name:
        st.session_state["monitor_batch"] = running.name if running else next(iter(by_name))
    selected = st.selectbox("Experiment history", list(by_name), key="monitor_batch",
                            format_func=lambda n: f"{by_name[n]['state'].upper()} · {n}")
    directory = by_name[selected]["directory"]
    st.query_params["batch"] = selected
    st.subheader(f"Experiment {selected}")
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "not_started"}
    plan = json.loads((directory / "plan.json").read_text())
    tasks = status.get("tasks", {})
    complete = sum(t["state"] == "completed" for t in tasks.values())
    total = len(plan["tasks"])
    st.progress(complete / max(1, total), text=f"{status['state']} · {complete}/{total} completed")
    if status.get("error"):
        st.error(status["error"])
    if status.get("current_model"):
        st.write(f"Current / last model: **{status['current_model']}**")
    active = running == directory
    if status["state"] == "running" and not process_alive(status.get("pid")):
        st.warning("Worker is no longer running. Completed outputs remain available; resume to retry remaining tasks.")
    if active and complete == 0:
        st.info("The worker is active. Loading a large model or generating its first answer can take time; "
                "0 completed questions does not mean nothing is happening. The log below updates every 3 seconds.")
    if status["state"] == "starting":
        st.info("Starting the worker. The experiment slot is already reserved; duplicate starts are blocked.")
    if status["state"] == "blocked":
        st.info("This is an older saved plan, not the currently running experiment. "
                "It never started because another experiment was running. It has not been lost. "
                "You can run it after the active experiment finishes or stops.")
    if (directory / "STOP").exists() and active:
        st.warning("Stopping requested. Waiting for the current model subprocess to exit...")
    c1, c2 = st.columns(2)
    if c1.button("Stop batch", disabled=not active):
        stop_clicked(directory)
        st.rerun(scope="app")
    if c2.button("Resume / retry incomplete", disabled=bool(running)):
        try:
            start_worker(directory)
            focus_batch(directory)
            st.rerun(scope="app")
        except BatchBusy as e:
            if e.directory:
                focus_batch(e.directory)
                st.rerun(scope="app")
            st.warning(str(e))
        except (OSError, ValueError) as e:
            st.error(str(e))
    st.dataframe(pd.DataFrame([{"model": t["model"]["name"], "prompt": t["prompt"]["id"],
                                **tasks.get(t["id"], {"state": "pending"})} for t in plan["tasks"]]),
                 hide_index=True, width="stretch")
    if status.get("current_log"):
        with st.expander("Live runner log", expanded=True):
            st.code(tail(status["current_log"]), language="text")
    with st.expander("Worker log"):
        st.code(tail(directory / "worker.log"), language="text")


with jobs:
    batch_monitor()
