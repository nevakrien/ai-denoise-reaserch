# Linear versus monotonic depth trends

Run `.venv/bin/python scripts/analyze_trends.py` to export
`results/depth-trends.csv`. The dashboard's **Compare models → Depth trend fits**
expander shows the same fits for the selected data. Configuration series remain
separate; runs are pooled with token weights, and each observed depth receives
equal weight in the regression. The script analyzes all saved runs, whereas the
dashboard respects batch/category/model filters. No new inference is required.

## Mistral smoke-run findings

`results/mistral-smoke.json` scores **two tokens**, not a substantial evaluation
corpus. The 32 layers are repeated measurements of those same two targets.

| Metric | Linear R² | Spearman rho | Isotonic R² | Single-step R² |
| --- | ---: | ---: | ---: | ---: |
| Perplexity | 0.582 | −0.911 | 0.814 | 0.574 |
| NLL = log(perplexity) | 0.791 | −0.911 | 0.974 | 0.687 |
| Top-1 agreement | 0.301 | +0.574 | 1.000 | 0.879 |

Agreement is 0 through layer 28, 0.5 at layers 29–31, and 1 at layer 32.
With two tokens, one changed argmax produces a **50 percentage-point jump**.
That staircase is not evidence of a broad, abrupt performance transition.

NLL has 11 wrong-way transitions out of 31. The largest increase is from
4.791 to 6.977 nats/token at layers 22→23. Nevertheless, its overall downward
rank trend is strong. Excluding the final layer gives NLL linear R² = 0.780,
rho = −0.902, and isotonic R² = 0.967: the trend is not solely a final-layer
artifact in this tiny sample. Agreement without the final layer is exactly a
single step (R² = 1), illustrating why high fit alone is not sufficient evidence.

## What the methods say

- **Linear regression:** measures straight-line fit. Fit NLL as well as raw PPL:
  exponentiation can turn a roughly linear NLL trend into a strongly curved PPL
  trend. Raw PPL puts enormous squared-error weight on early high values.
- **Spearman correlation:** measures ordering, not linearity. Negative values
  support decreasing PPL/NLL; positive values support increasing agreement.
  Ties are assigned average ranks. Constant responses have undefined rho/R².
- **Isotonic regression:** least-squares fit constrained to decrease for NLL/PPL
  and increase for agreement. It allows plateaus, steps, and nonlinear shapes.
  Wrong-way-transition counts describe departures in the *observations*; the
  fitted curve is monotonic by construction. Its high in-sample R² is not a test
  that the underlying process is monotonic.
- **Single-step fit:** fits two constant segments, scanning all possible splits.
  This checks whether one jump can describe the curve without a gradual trend.
  Its breakpoint is selected on the same data, not independently confirmed.

## Adjacent-layer NLL changes and the coin-flip reference

The main comparison UI also reports ΔNLL = NLL(i+1) − NLL(i) for each
adjacent pair in each token-weighted model/configuration curve. Negative means
improved, positive means worsened, and only exactly zero changes are
ties. No tolerance is applied. Down/up/tied percentages use **all** transitions as the denominator.
The summary and individual deltas can be downloaded as CSV. Filters apply to
the underlying data; chart zoom does not. A separate checkbox excludes the
final layer and its incoming transition.

The exact two-sided coin-flip test compares **improved (ΔNLL < 0)** against
**not improved (ΔNLL ≥ 0)**. Exact zeros count as not improved, not excluded.
With k improvements among n total transitions, its null is independent fair
improved/not-improved outcomes and its alternative
is improvement probability ≠ 0.5:

`p_better = P[Binomial(n, 0.5) >= k]`

`p_not_improved = P[Binomial(n, 0.5) >= n-k]`

`p_two_sided = min(1, 2 * min(p_better, p_not_improved))`.

The UI's **signed p** is positive when improvements outnumber not-improved
transitions and negative for the opposite majority. Its magnitude is the ordinary
two-sided p-value; it is a direction-annotated display, not a negative
probability. Balanced counts are labeled balanced. Every negative signed p
(not-improved majority) receives a bright-red row, regardless of significance. Non-negative
signed p < 0.05 receives a green row; red takes priority when both conditions
apply. All other results are yellow (inconclusive), including balanced counts
and undefined p-values when there are no transitions. More data may help
resolve inconclusive results but does not guarantee significance.
Color alone indicates the result; no significance checkbox is displayed.
CSV summaries include both
directional tails, the two-sided p-value, and the signed display value.

No transitions means an undefined p-value, not evidence either way.
This tests direction frequency, not the sizes of improvements or total NLL
reduction. Each aggregate layer transition contributes one sign, not one sign
per underlying token. Results are unadjusted for multiple model comparisons.

The null is explicitly **independent fair-coin signs**. A small p-value means
an imbalance at least this extreme would be unlikely under that null. The test
detects an excess of either improved or not-improved outcomes; it does not distinguish a directional
bias from dependence between signs. Dependence of layer outputs alone does
not establish dependence of their change signs. Testing serial dependence
itself would require a separate statistic, such as a runs statistic.

These models have different flexibility, so their training R² values are **not**
a fair predictive model-selection test. No p-values are reported for these fits: adjacent
layers and tokens are dependent, and two tokens provide no useful independent
replication. A monotonicity test such as Mann–Kendall also requires care with
dependence; applying an off-the-shelf independent-observation test to layers
would not fix the experimental design.

## Alternative explanations and next experiment

1. **Target selection:** targets are the full model's greedy outputs. Its final
   argmax therefore agrees by construction (up to numerical differences), and
   intermediate layers are being measured for convergence to that output, not
   independent correctness. Excluding the final layer helps diagnose endpoint
   leverage but does not remove target-selection bias.
2. **Readout alignment:** every layer uses the final normalization and output
   head. Early representations may contain information that this fixed readout
   cannot decode. Improving logit-lens scores need not mean that equivalent
   amounts of useful information were absent earlier. A trained, independently
   validated per-layer readout is a useful control.
3. **Discrete argmax and exponential PPL:** tiny changes across a top-1 boundary
   can jump agreement, while exponential PPL magnifies NLL changes. NLL and
   target probabilities are more informative about smooth changes.
4. **Small or mixed samples:** a few tokens, prompt type, special tokens, and
   aggregation can hide token-level reversals. Inspect per-prompt curves and
   token/layer heatmaps; do not interpret a pooled trend as universal.

Next, score many fixed **external** continuations across independent documents
and prompt categories, using the same targets at each depth. Separate training
and held-out documents if fitting readouts or selecting shapes/change points.
Validate linear/isotonic/step predictions on held-out documents, and bootstrap
whole documents/prompts (not individual layers) for uncertainty. Stratify token
types and check prompt-level trends. Independently generated early-exit answers
are needed for claims about usable answer quality. None of these fits alone
establish a causal mechanism or diffusion-like computation.
