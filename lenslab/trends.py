"""Depth trends and an explicitly independent-coin-flip reference test."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def nll_result_style(signed_p):
    """Not-improved majority takes priority; inconclusive results stay yellow."""
    if signed_p < 0:
        return "background-color: #ff2020; color: #ffffff; font-weight: bold"
    if signed_p < .05:
        return "background-color: #28a745; color: #ffffff; font-weight: bold"
    return "background-color: #ffeb3b; color: #000000; font-weight: bold"


def nll_change_stats(values):
    """Exact two-sided coin-flip test of improved versus not improved."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("NLL values must be a finite one-dimensional sequence")
    delta = np.diff(values)
    down = int(np.sum(delta < 0))
    up = int(np.sum(delta > 0))
    tied = len(delta) - down - up
    n = len(delta)
    not_improved = up + tied
    # P[X >= observed improvements], X ~ Binomial(n, 0.5).
    p = sum(math.comb(n, k) for k in range(down, n + 1)) / (2 ** n) if n else np.nan
    not_improved_p = sum(math.comb(n, k) for k in range(not_improved, n + 1)) / (2 ** n) if n else np.nan
    two_sided = min(1.0, 2 * min(p, not_improved_p)) if n else np.nan
    direction = "improvement" if down > not_improved else "not improved" if not_improved > down else "balanced" if n else "no transitions"
    signed_p = -two_sided if not_improved > down else two_sided
    return {"transitions": len(delta), "improved": down, "worsened": up, "tied": tied,
            "improved_pct": 100 * down / len(delta) if len(delta) else np.nan,
            "worsened_pct": 100 * up / len(delta) if len(delta) else np.nan,
            "tied_pct": 100 * tied / len(delta) if len(delta) else np.nan,
            "not_improved": not_improved,
            "not_improved_pct": 100 * not_improved / n if n else np.nan,
            "non_tied": down + up, "coin_flip_p_one_sided": p,
            "coin_flip_p_not_improved": not_improved_p, "coin_flip_p_two_sided": two_sided,
            "direction": direction, "signed_p_display": signed_p,
            "mean_delta_nll": float(delta.mean()) if len(delta) else np.nan}


def nll_transition_tables(aggregated, exclude_final=False):
    """One coin flip per adjacent transition of each token-weighted model curve."""
    summaries, transitions = [], []
    if aggregated.empty:
        return pd.DataFrame(), pd.DataFrame()
    for series, group in aggregated.groupby("series", sort=False):
        group = group.sort_values("kept")
        if exclude_final:
            group = group.iloc[:-1]
        if group.empty:
            continue
        if np.any(np.diff(group["kept"]) != 1):
            raise ValueError("NLL transitions require consecutive layers")
        meta = {"series": series, "model": group.iloc[0]["model"],
                "configuration": group.iloc[0]["configuration"],
                "scope": "excluding final layer" if exclude_final else "all layers"}
        summaries.append({**meta, **nll_change_stats(group["nll"])})
        for i in range(1, len(group)):
            before, after = group.iloc[i - 1], group.iloc[i]
            delta = float(after["nll"] - before["nll"])
            transitions.append({**meta, "from_layer": int(before["kept"]),
                                "to_layer": int(after["kept"]), "nll_before": before["nll"],
                                "nll_after": after["nll"], "delta_nll": delta,
                                "test_outcome": "improved" if delta < 0 else "not improved",
                                "direction": "improved" if delta < 0 else
                                "worsened" if delta > 0 else "tied"})
    return pd.DataFrame(summaries), pd.DataFrame(transitions)


def isotonic(values, increasing=True):
    """Equal-depth least-squares isotonic fit via pool-adjacent-violators."""
    sign = 1 if increasing else -1
    blocks = []
    for value in np.asarray(values, dtype=float) * sign:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    return sign * np.repeat([b[0] for b in blocks], [b[1] for b in blocks])


def r_squared(y, fitted):
    total = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - fitted) ** 2) / total) if total > 0 else np.nan


def fit_trend(depth, values, increasing):
    x, y = np.asarray(depth, dtype=float), np.asarray(values, dtype=float)
    if len(x) < 3 or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError("Need at least three ordered, distinct depths and finite values")
    slope, intercept = np.polyfit(x, y, 1)
    linear = intercept + slope * x
    monotone = isotonic(y, increasing)
    # Exhaustive descriptive change-point fit: constant on either side.
    candidates = [np.r_[np.repeat(y[:k].mean(), k), np.repeat(y[k:].mean(), len(y) - k)]
                  for k in range(1, len(y))]
    split = int(np.argmin([np.sum((y - fit) ** 2) for fit in candidates])) + 1
    step = candidates[split - 1]
    delta = np.diff(y)
    jump = int(np.argmax(np.abs(delta)))
    ranks = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(x.argsort().argsort(), ranks)[0, 1]) if np.ptp(y) else np.nan
    stats = {"linear_R2": r_squared(y, linear), "slope_per_full_depth": float(slope),
             "spearman_rho": rho, "isotonic_R2": r_squared(y, monotone),
             "single_step_R2": r_squared(y, step), "step_starts_depth": float(x[split]),
             "wrong_way_transitions": int(np.sum(delta < -1e-6 if increasing else delta > 1e-6)),
             "transitions": len(delta), "largest_jump_to_depth": float(x[jump + 1]),
             "largest_jump": float(delta[jump])}
    return stats, {"linear": linear, "isotonic": monotone, "single step": step}


def trend_summary(aggregated):
    rows = []
    if aggregated.empty:
        return pd.DataFrame()
    for series, group in aggregated.groupby("series", sort=False):
        group = group.sort_values("kept")
        for scope in ("all layers", "excluding final layer"):
            data = group if scope == "all layers" else group.iloc[:-1]
            if len(data) < 3:
                continue
            for metric in ("perplexity", "nll", "agreement"):
                stats, _ = fit_trend(data["depth"], data[metric], metric == "agreement")
                rows.append({"series": series, "model": group.iloc[0]["model"],
                             "configuration": group.iloc[0]["configuration"], "scope": scope,
                             "metric": metric, "layers": len(data), **stats})
    return pd.DataFrame(rows)


def trend_figure(group, metric="nll", exclude_final=False):
    data = group.sort_values("kept")
    if exclude_final:
        data = data.iloc[:-1]
    _, fits = fit_trend(data["depth"], data[metric], metric == "agreement")
    fig = go.Figure(go.Scatter(x=data["depth"] * 100, y=data[metric], name="observed",
                              mode="lines+markers"))
    for name, values in fits.items():
        fig.add_trace(go.Scatter(x=data["depth"] * 100, y=values, name=name,
                                 mode="lines", line=dict(dash="dash")))
    fig.update_layout(template="plotly_white", xaxis_title="Transformer depth retained (%)",
                      yaxis_title={"nll": "NLL = log(perplexity) (nats/token)",
                                   "agreement": "Top-1 agreement", "perplexity": "Perplexity"}[metric])
    return fig
