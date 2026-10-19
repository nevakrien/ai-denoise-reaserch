# GGUF early-exit / logit-lens experiment

Hypothesis: intermediate transformer residual states become increasingly useful
for next-token prediction when decoded with the original final normalization and
output head. Removing more top layers may increase perplexity; monotonicity is
an experimental question, not a property imposed by this program.

This prototype uses **unmodified, pinned llama.cpp**, not the Ollama server.
It can read ordinary GGUF files or existing Ollama GGUF blobs directly.
No conversion to Hugging Face weights or full-precision output matrix is needed.

## Python dashboard

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt
.venv/bin/python -m streamlit run dashboard.py
```

Open **http://127.0.0.1:8501**. The dashboard provides:

- Local model discovery, including Ollama blobs and split GGUFs; unsupported
  architectures and missing shards are reported rather than silently measured.
- Model selection, editable GPU-layer budgets, all available CPU threads by
  default, token budgets, custom questions, and a 12-question categorized suite.
- Persistent background batches, live logs, stop/resume, and cached completed
  tasks. Each model loads once for all selected prompts; models run sequentially.
- Token-weighted per-model depth curves, log-scale perplexity, native-relative
  excess NLL and PPL ratios, agreement plots, token/layer heatmaps, answer and
  teacher-forced prediction inspection, CSV/JSON and interactive HTML exports.

Collection survives browser refreshes and server shutdowns. Use **Batch monitor**
to stop the worker; closing the browser does not stop it. Click **Refresh results**
to update analysis during collection. The server binds to localhost and disables
usage telemetry. This is a trusted local tool, not a multi-user service.

Detailed controls and the larger-model pilot: [docs/experiment-lab.md](docs/experiment-lab.md).

## Build

Requires Git, CMake >= 3.19, and a C++17 compiler. The first configure downloads
llama.cpp at the revision recorded in `CMakeLists.txt`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To reuse a checkout at that revision, add
`-DFETCHCONTENT_SOURCE_DIR_LLAMA=/path/to/llama.cpp` to configure.
CPU is the default. For CUDA use a separate `build-cuda` directory with
`-DGGML_CUDA=ON`, then run with `--gpu-layers 99` (or a smaller number if the
weights do not fit). The native final norm/head and reconstructed head run on
CPU to match quantized arithmetic; transformer blocks can run on GPU.

## Ask a question

```bash
./build/gguf-lens --model /path/to/model.gguf \
  --prompt "Why is the sky blue?" --tokens 32 --threads 8 \
  --output results/sky.json --save-hidden
```

Or use a model already downloaded by Ollama (no running server required):

```bash
MODEL=$(python3 scripts/ollama_gguf.py llama3.1)
./build/gguf-lens --model "$MODEL" --prompt "Why is the sky blue?" \
  --tokens 32 --threads 8 --output results/sky.json --save-hidden
```

The helper respects `OLLAMA_MODELS`, or accepts `--root /path/to/models`.
It resolves the weights only: Ollama's separate system prompt, template, adapters,
and sampling configuration are **not** reproduced. The runner uses the GGUF chat
template. If missing or unsupported, pass an appropriately formatted prompt with
`--raw-prompt` (also useful for base models).

## Measurement

1. The complete quantized model generates an answer greedily.
2. At each prediction position, the evaluation callback captures the residual
   stream **after every transformer block**, before final normalization.
3. Each state is decoded as `output_weight @ final_RMSNorm(hidden_state)`.
   The output weight remains in its original GGML quantization. Tied heads work
   through llama.cpp's resolved output tensor.
4. All depths score the **same full-model-generated target token** with an
   untempered full-vocabulary softmax. Prompt tokens are not scored. A generated
   end-of-generation token is scored, if reached, but excluded from answer text.
5. `perplexity = exp(mean(-log p(target | prefix)))`, in natural-log units.
   There is no off-by-one shift: the last prompt state predicts the first answer
   token; the first answer state predicts the second token, and so on.

The terminal prints the full answer and a table by depth. JSON contains per-layer
perplexity, mean NLL, top-1 agreement, per-token scores/argmax IDs, exact prompt
IDs, settings, and the full-model baseline. Layer counts are **1-based**;
per-token arrays and hidden-state layer indices are **0-based**.

`teacher_forced_argmax_text` concatenates each layer's guesses under the full
model's prefixes. It is **not an independently generated early-exit answer**.
This initial experiment measures prediction degradation; it does not yet test
free-running early-exit generation or inference speedups. All blocks still run.

### Hidden states

`--save-hidden` writes `OUTPUT.json.hidden.f32`, a native-endian float32 array
with shape `[scored_tokens, transformer_layers, hidden_width]` in C order.
The JSON records dimensions. These are the states **before** the scored target,
not the states after consuming it. Only answer-prediction positions are saved.

```python
import json
import numpy as np

run = json.load(open("results/sky.json"))
meta = run["hidden_states"]
hidden = np.memmap(meta["path"], dtype=np.float32, mode="r",
                   shape=tuple(meta["shape"]))
print(hidden.shape)
```

## Support and correctness limits

- Architecture adapters: GGUF `llama` (including Mistral/Mixtral), `qwen2`,
  `qwen3`, `qwen2moe`, `qwen3moe`, `qwen35`, `qwen35moe`, `glm4moe`, `glm-dsa`,
  and `glm5-next`. GLM-5.3-Flash's four streams are averaged before final RMSNorm,
  matching its native head. Every run must pass the final-layer check; an adapter
  entry alone does not mean every model in that family has been tested.
  Gemma, GPT-OSS, and other head conventions still need separate adapters.
- A final-layer sanity check compares reconstructed versus native target-token
  NLL at every position. A difference above 0.1 nat aborts; the maximum observed
  difference is recorded. Quantized CPU/GPU matmul paths can differ numerically;
  this check is a guardrail, not proof of exact logit equivalence.
- Hidden activations are copied to CPU. A CPU copy of the quantized head and
  `[layers, vocabulary]` float logits are allocated. Large heads/vocabularies can
  consume significant RAM, and projecting every layer is expensive.
- For multi-stream models, hidden-state shape is
  `[scored_tokens, layers, residual_streams, hidden_width]`. Ordinary single-stream
  models retain the original three-dimensional layout. Raw streams, not their
  mean, are saved.
- Self-generated greedy-answer perplexity is biased toward the full model and
  is **not held-out language-model perplexity**. Use it only as a smoke test.
  A future experiment should score a fixed external corpus and measure independent
  early-exit continuations. Improving predictions across layers alone does not
  establish that transformers implement diffusion.
- Token pieces may split UTF-8 characters; exact token IDs are authoritative.
  Invalid UTF-8 in JSON display strings is replaced, not in hidden data or IDs.

Python helper tests (optional `pytest` dependency):

```bash
python3 -m pytest tests/test_ollama_gguf.py
```

Validate the metrics and hidden-file layout of an actual run:

```bash
GGUF_LENS_RESULT=results/sky.json python3 -m pytest tests -q
```

See [the initial smoke test](docs/initial-smoke-test.md) for measured results.
