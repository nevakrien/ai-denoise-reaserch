# Experiment lab

## Available locally

The discovery pass found Llama 3/3.1 8B and 70B, Mistral, Mixtral, Qwen3 8B,
Qwen3-Coder 30B MoE, DeepSeek-R1 distilled 32B/70B, Qwen3.5-27B, and
GLM-5.3-Flash, among others. No model weights were downloaded.

The two specifically requested files are:

- `/media/user/archive/llamacpp/models/Qwen3.5-27B/Qwen3.5-27B-UD-Q4_K_XL.gguf`
  (16.41 GiB, 64 hybrid blocks).
- `/media/user/archive/llamacpp/models/GLM-5.3-Flash/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf`
  plus its second shard (102.47 GiB total, 45 hybrid/MoE blocks, four residual
  streams). Select the first shard only; llama.cpp loads both.

`config/models.local.json` is an ignored, machine-specific catalog. Discovery
reads metadata without loading weights into the inference runtime. It groups
shards, checks their existence, deduplicates identical Ollama blobs, and records
architecture, block count, width, quantization metadata, and total file size.
If the GGUF Python reader cannot inspect a quantization, the error is displayed.
Models with no adapter are not selectable for collection.

## Prompts

`config/prompts.json` contains 12 editable questions: causal inference, Bayesian
arithmetic, Python debugging, algorithm design, false premises, Hebrew science
explanation, constrained storytelling, structured extraction, counterfactual
physics, invariant proofs, calibrated skepticism, and ambiguous translation.

Choose any subset in the UI, edit the text, or add a custom question. The exact
question and formatted token IDs are saved with every measurement. A 48-token
pilot is intentionally short: answers can be truncated, and reasoning models may
spend the whole budget on reasoning rather than their final answer. Use longer
budgets when studying answer content, and compare the same prompt subset across
models. The GGUF's chat template is passed to llama.cpp's supported-template
formatter, not a general Jinja interpreter. Missing/unsupported templates fail
explicitly; raw mode requires you to supply appropriate formatting yourself.

## Current broad pilot

The initial survey selects eight models and six questions (48 runs):

- GLM-5.3-Flash, Qwen3.5-27B, Llama 3.1 8B, Qwen3 8B,
  Qwen3-Coder 30B, DeepSeek-R1 32B, Mixtral, and Llama 3.1 70B.
- Causal inference, Bayes, code debugging, false-premise correction, Hebrew,
  and constrained fiction.
- 48 generated tokens, 32 CPU threads, CUDA transformer offload with an
  approximate 18 GiB weight budget; final norm/head on CPU.

The batch monitor and on-disk status, **not this document**, are authoritative
about completion and failures. Large models on the archive HDD can take a long
time to load. Avoid treating an incomplete pilot as an evaluation of all eight
models. Completed older smoke-test/partial-survey outputs are retained separately.

An earlier partial CUDA survey found a prompt where native GPU-head versus
reconstructed CPU-head NLL differed by more than the 0.1-nat guardrail. The run
was rejected, not included in plots. The revised runner pins output normalization,
output weights, and tied embeddings to CPU while allowing GPU transformer blocks.
This keeps native and reconstructed head arithmetic comparable. The head copy
retains its original GGML quantization; it is not dequantized to FP32.

For GLM-5.3-Flash, each captured state is `[hidden_width, 4]`. The exit head
averages the four streams, then applies the original final RMSNorm and output
matrix. This reproduces the architecture's `build_hc_mean` convention rather
than arbitrarily choosing a stream. The last block must still pass the NLL
check. This tests block-depth truncation in a hybrid model, not a claim that its
blocks are all conventional attention layers.

## Running without the UI

```bash
.venv/bin/python -m lenslab discover /media/user/archive/llamacpp/models
.venv/bin/python -m lenslab run --models MODEL_ID ANOTHER_MODEL_ID \
  --prompts bayes code-debug false-premise \
  --tokens 96 --binary build-cuda/gguf-lens --gpu-budget-gib 18
```

Omitting `--threads` uses all CPU threads available to the Python process (32 on
this machine). Both CPU and CUDA builds use full build parallelism:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda -j
```

A complete CUDA development toolkit is required, not just an NVIDIA driver.
This machine's existing Conda compiler lacked runtime/development headers; local
CUDA wheel headers/libraries were assembled under the ignored `.deps/cuda` for
the working build. These additions are confined to the project `.venv`/`.deps`;
the global CUDA installation was not changed. The existing working CMake cache
records that local toolkit. On a machine with a standard toolkit, the commands
above suffice.

### Direct C++ multi-question mode

`gguf-lens --model PATH --requests FILE` accepts a JSON array:

```json
[
  {"prompt": "Explain Bayes' rule.", "output": "results/bayes.json"},
  {"prompt": "Write a short story.", "output": "results/story.json", "raw_prompt": false}
]
```

The model/head load once; each question gets a fresh context and cache. Errors
are reported per question, successful outputs remain, and the process returns
2 if any question failed. Shared CLI settings apply to all questions.

## Persistence, resource control, and resume

- Start immediately reserves the experiment slot and opens that exact experiment's
  monitor. Duplicate clicks and simultaneous browser submissions cannot create
  a second active batch. While a worker is starting or running, Start is disabled.
- A persistent active-experiment banner provides Open/Stop controls on every tab.
  The monitor remembers its selected experiment; the URL records the selected tab
  and batch so refreshes and copied links return to it.
- Experiment history includes blocked/not-started saved plans. A blocked plan did
  not run and is not lost; it can be started after the active experiment stops.
- `results/lab/runs/TASK_ID.json`: completed measurements and provenance.
- `results/lab/batches/BATCH_ID/plan.json`: immutable model/question/settings plan.
- `status.json`: atomically updated task states; model logs retain full errors.
- Startup reservation and a worker lock prevent concurrent batches from competing
  for RAM/VRAM, including the gap before a newly launched worker has initialized.
- Stop creates a flag; the worker terminates the current model subprocess group,
  preserves completed questions, and marks remaining tasks cancelled.
- Resume skips validated outputs and reloads a model only for incomplete questions.
  It reuses tasks with matching prompt text, settings, model paths/sizes/mtimes,
  and executable SHA-256. Model fingerprints are not content hashes of all weights.
- After rebuilding the executable or changing model files, create a **new** batch.
  Old plans are rejected rather than silently mixing versions.
- GPU layer allocation is a conservative size-based estimate, not a memory
  guarantee. Override it per model. A memory allocation error is visible in the
  log; retry in a new batch with fewer GPU layers. More CPU threads do not remove
  RAM, VRAM, or disk-bandwidth bottlenecks.

## Statistics

The UI keeps model/configuration series separate. Different token budgets,
executable versions, and GPU placements are not silently pooled into one curve.
Within a series, aggregate mean NLL is weighted by the number of scored tokens;
aggregate perplexity is `exp(pooled_mean_nll)`, **not** an arithmetic mean of PPLs.
Per-prompt lines expose variability; they are not confidence intervals.

Primary cross-model plot: **excess NLL versus fraction of blocks retained**.
Also available: log PPL, PPL ratio to native baseline, target top-1 agreement,
and token-by-layer signed excess-NLL heatmaps. Hover shows exact depth/token
information. Negative excess NLL at some positions is allowed; intermediate
layers can assign higher probability to a particular target than the last layer.

These measures remain biased by self-generation. Tokenizers, generated content,
and vocabularies differ across models, so even normalized curves are not a fair
language-model-quality ranking. A proper follow-up needs a fixed external corpus,
more prompt samples, and independent free-running early exits. Hidden-state
inspection and teacher-forced guesses alone do not establish legible autonomous
early-exit generation or diffusion-like computation.

## Tests

```bash
ctest --test-dir build-cuda --output-on-failure
.venv/bin/python -m pytest tests -q
GGUF_LENS_RESULT=/absolute/path/to/run.json .venv/bin/python -m pytest tests/test_saved_result.py -q
```

Tests cover weighted aggregation, corrupt-result exclusion, plotting, batch
model reuse, cache/provenance, cancellation, failed questions, binary changes,
and the Ollama resolver. Actual model runs additionally enforce the native-final
NLL check. GPU/architecture availability is not inferred from unit-test success.
