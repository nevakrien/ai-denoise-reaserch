# Initial smoke test

Model: Meta Llama 3.1 8B Instruct, cached Ollama `llama3.1:latest` GGUF.
The file reports Q4_0; the output head is Q6_K.
Blob SHA-256: `87048bcd55216712ef14c11c2c303728463207b165bf18440b9b84b07ec00f87`.
CPU execution, 8 threads, 32 generated tokens, embedded GGUF chat template.
llama.cpp revision: `7951f601e89494f2ed12dbd4eb52e254b8401197`.

Question: **Why is the sky blue?**

Generated answer (truncated at the token budget):

> The sky appears blue to us because of a phenomenon called scattering, which occurs when sunlight interacts with the tiny molecules of gases in the atmosphere. Here's a simplified

All depths scored these same 32 generated tokens:

| Layers kept | Layers cut | Perplexity | Top-1 agreement |
| ---: | ---: | ---: | ---: |
| 16 | 16 | 37,714.8 | 0% |
| 20 | 12 | 587.72 | 31.25% |
| 24 | 8 | 90.43 | 34.375% |
| 28 | 4 | 10.63 | 62.5% |
| 30 | 2 | 3.84 | 62.5% |
| 31 | 1 | 1.81 | 81.25% |
| 32 | 0 | 1.12623 | 100% |

Maximum final-layer versus native target-token NLL difference:
**1.67e-6 nat**. Saved hidden-state shape: **[32, 32, 4096]**.
JSON and binary output are in the ignored `results/llama31-sky.json` and
`results/llama31-sky.json.hidden.f32` files on the test machine.

Perplexity generally improves with depth, but not monotonically. It increases
from layer 20 to 21 (587.72 to 593.65), for example. Half-depth predictions are
poor with the unmodified final head in this run. This does not exclude useful
intermediate representations with a learned/tuned lens.

These numbers are a functional check, not a corpus evaluation. Full-model greedy
generation selects high-probability full-model tokens, strongly favoring that
baseline. This test neither establishes legible free-running early-exit output
nor demonstrates diffusion-like computation.

## Additional adapter checks

Two-token runs also completed on the cached Mistral and Qwen3 8B models:

| Model | GGUF architecture | Maximum final-layer NLL error |
| --- | --- | ---: |
| Mistral | llama | 1.75e-8 nat |
| Qwen3 8B | qwen3 | 1.63e-12 nat |

Both used Q6_K output heads. These are adapter checks, not quality evaluations;
Qwen3's two generated tokens were just the opening reasoning marker. Qwen2 is
implemented from its matching graph convention but has not been tested with a
local model. GPU execution has not been tested.

Validation passed: 2 CTest tests, plus 7 Python checks for each saved run
(resolver behavior, recomputed metrics, token alignment, hidden layout/finite values).
