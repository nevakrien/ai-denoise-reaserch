import numpy as np
import pytest

from lenslab.trends import (fit_trend, isotonic, nll_change_stats, nll_result_style, nll_transition_tables,
                            trend_figure, trend_summary)


@pytest.mark.parametrize("p,color", [(.001, "#28a745"), (.049, "#28a745"),
                                   (.05, "#ffeb3b"), (.8, "#ffeb3b"), (1, "#ffeb3b"),
                                   (np.nan, "#ffeb3b"), (-.8, "#ff2020"), (-.001, "#ff2020")])
def test_nll_result_color_rules(p, color):
    assert color in nll_result_style(p)


def test_nll_coin_flip_sign_test():
    stats = nll_change_stats([5, 4, 3, 2, 1])
    assert stats["improved"] == 4
    assert stats["improved_pct"] == 100
    assert stats["coin_flip_p_one_sided"] == 1 / 16
    assert stats["mean_delta_nll"] == -1
    assert nll_change_stats([1, 2, 3])["coin_flip_p_one_sided"] == 1
    # Three improvements and one worsening: P[Binomial(4, .5) >= 3] = 5/16.
    stats = nll_change_stats([5, 4, 3, 4, 3])
    assert stats["coin_flip_p_one_sided"] == 5 / 16
    assert stats["improved_pct"] == 75
    assert stats["worsened_pct"] == 25


def test_two_sided_test_detects_both_directions_symmetrically():
    better = nll_change_stats(list(range(10, 0, -1)))
    worse = nll_change_stats(list(range(10)))
    assert better["coin_flip_p_two_sided"] == worse["coin_flip_p_two_sided"] == 2 / 512
    assert better["signed_p_display"] == 2 / 512
    assert worse["signed_p_display"] == -2 / 512
    assert worse["direction"] == "not improved"
    balanced = nll_change_stats([1, 2, 1])
    assert balanced["coin_flip_p_two_sided"] == 1
    assert balanced["direction"] == "balanced"
    assert nll_change_stats([1, 1])["signed_p_display"] == -1


def test_nll_ties_and_degenerate_data():
    stats = nll_change_stats([3, 2, 2, 1])
    assert stats["tied"] == 1
    assert stats["non_tied"] == 2
    assert stats["not_improved"] == 1
    assert stats["coin_flip_p_one_sided"] == .5
    assert sum(stats[k] for k in ("improved_pct", "worsened_pct", "tied_pct")) == pytest.approx(100)
    assert nll_change_stats([1, 1, 1])["coin_flip_p_one_sided"] == 1
    assert nll_change_stats([1, 1, 1])["signed_p_display"] == -.5
    assert nll_change_stats([1] * 10)["signed_p_display"] == -2 / 512
    assert np.isnan(nll_change_stats([1])["improved_pct"])
    assert nll_change_stats([])["transitions"] == 0
    with pytest.raises(ValueError):
        nll_change_stats([1, np.nan])


def test_nll_changes_use_strict_sign_without_tolerance():
    stats = nll_change_stats([1, 1 + 1e-12, 1, 1])
    assert stats["improved"] == 1
    assert stats["worsened"] == 1
    assert stats["tied"] == 1
    assert stats["non_tied"] == 2
    assert stats["not_improved"] == 2
    assert stats["coin_flip_p_one_sided"] == .875


def test_nll_transition_tables_sort_and_separate_series():
    import pandas as pd

    data = pd.DataFrame({"series": ["s"] * 3 + ["t"] * 3, "model": ["m"] * 6,
                         "configuration": ["c"] * 6, "kept": [3, 1, 2, 1, 2, 3],
                         "nll": [1, 3, 2, 1, 2, 3]})
    summary, details = nll_transition_tables(data)
    assert summary.improved.tolist() == [2, 0]
    assert summary.coin_flip_p_one_sided.tolist() == [.25, 1]
    assert details.delta_nll.tolist() == [-1, -1, 1, 1]
    assert details.from_layer.tolist() == [1, 2, 1, 2]
    summary, details = nll_transition_tables(data, exclude_final=True)
    assert summary.transitions.tolist() == [1, 1]
    assert summary.coin_flip_p_one_sided.tolist() == [.5, 1]
    assert len(details) == 2
    assert all(frame.empty for frame in nll_transition_tables(pd.DataFrame()))
    with pytest.raises(ValueError, match="consecutive"):
        nll_transition_tables(data.iloc[:2])


def test_pava_pools_violations():
    assert isotonic([3, 1, 2, 4]).tolist() == [2, 2, 2, 4]
    assert isotonic([4, 2, 3, 1], increasing=False).tolist() == [4, 2.5, 2.5, 1]


def test_linear_trend():
    stats, fits = fit_trend([0.25, 0.5, 0.75, 1], [4, 3, 2, 1], False)
    assert stats["linear_R2"] == pytest.approx(1)
    assert stats["spearman_rho"] == pytest.approx(-1)
    assert stats["isotonic_R2"] == 1
    assert stats["wrong_way_transitions"] == 0
    assert np.allclose(fits["linear"], [4, 3, 2, 1])


def test_step_and_constant():
    stats, _ = fit_trend(np.arange(6), [0, 0, 0, 1, 1, 1], True)
    assert stats["single_step_R2"] == 1
    assert stats["step_starts_depth"] == 3
    assert stats["linear_R2"] < stats["isotonic_R2"]
    stats, _ = fit_trend(np.arange(4), [1] * 4, True)
    assert np.isnan(stats["linear_R2"])
    assert np.isnan(stats["spearman_rho"])


def test_summary_and_final_layer_sensitivity():
    import pandas as pd

    data = pd.DataFrame({"series": ["s"] * 4, "model": ["m"] * 4,
                         "configuration": ["c"] * 4, "kept": [1, 2, 3, 4],
                         "depth": [.25, .5, .75, 1], "nll": [3, 3, 3, 0],
                         "perplexity": np.exp([3, 3, 3, 0]), "agreement": [0, 0, 0, 1]})
    summary = trend_summary(data)
    assert len(summary) == 6
    assert summary[summary.scope == "excluding final layer"].linear_R2.isna().all()
    assert len(trend_figure(data).data) == 4
    assert trend_summary(pd.DataFrame()).empty


def test_invalid_depths():
    with pytest.raises(ValueError):
        fit_trend([0, 0, 1], [1, 2, 3], True)
