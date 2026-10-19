#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include "metrics.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

struct options {
    std::string model, prompt = "Why is the sky blue?", output = "results/run.json";
    int tokens = 32, threads = std::max(1u, std::thread::hardware_concurrency() / 2), gpu_layers = 0;
    bool raw = false, save_hidden = false;
};

static void usage() {
    std::cout << "gguf-lens --model PATH [--prompt QUESTION] [--tokens 32]\n"
                 "          [--output results/run.json] [--threads N] [--gpu-layers N]\n"
                 "          [--raw-prompt] [--save-hidden]\n"
                 "Full-model greedy generation; every layer scores the same generated tokens.\n";
}

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--raw-prompt") { o.raw = true; continue; }
        if (key == "--save-hidden") { o.save_hidden = true; continue; }
        if (i + 1 == argc) { throw std::runtime_error("Missing value for " + key); }
        std::string value = argv[++i];
        if (key == "--model") o.model = value;
        else if (key == "--prompt") o.prompt = value;
        else if (key == "--output") o.output = value;
        else if (key == "--tokens") o.tokens = std::stoi(value);
        else if (key == "--threads") o.threads = std::stoi(value);
        else if (key == "--gpu-layers") o.gpu_layers = std::stoi(value);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (o.model.empty() || o.prompt.empty() || o.output.empty() || o.tokens <= 0 || o.threads <= 0) {
        throw std::runtime_error("Model, nonempty prompt/output, and positive tokens/threads required");
    }
    return o;
}

struct capture {
    int width, layers;
    bool active = false;
    std::vector<float> states;
    std::vector<bool> seen;
    std::string error;

    capture(int w, int l) : width(w), layers(l), states(size_t(w) * l), seen(l, false) {}
    void reset() { std::fill(seen.begin(), seen.end(), false); error.clear(); }

    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & c = *static_cast<capture *>(data);
        if (!c.active) return false;
        const std::string name = t->name;
        if (name.rfind("l_out-", 0) != 0) return false;
        if (ask) return true;
        // Do not let exceptions cross the C callback boundary.
        try {
            size_t consumed = 0;
            int layer = std::stoi(name.substr(6), &consumed);
            if (consumed != name.size() - 6 || layer < 0 || layer >= c.layers ||
                t->type != GGML_TYPE_F32 || t->ne[0] != c.width ||
                t->ne[1] < 1 || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t)) {
                throw std::runtime_error("Unexpected hidden tensor: " + name);
            }
            size_t offset = size_t(t->ne[1] - 1) * t->nb[1];
            ggml_backend_tensor_get(t, c.states.data() + size_t(layer) * c.width,
                                    offset, size_t(c.width) * sizeof(float));
            c.seen[layer] = true;
        } catch (const std::exception & e) { c.error = e.what(); }
        return true;
    }

    void check() const {
        if (!error.empty()) throw std::runtime_error(error);
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
            throw std::runtime_error("Missing layer tensors; llama.cpp graph does not match this adapter");
        }
    }
};

// Reuse GGML's quantized matmul; no Python dequantization or full FP32 head copy.
struct head {
    std::vector<uint8_t> arena, work;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    ggml_tensor * input = nullptr, * logits = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};

    head(const llama_model & model, int width, int layers, int vocab, int threads) {
        size_t bytes = ggml_nbytes(model.output) + ggml_nbytes(model.output_norm) +
                       size_t(width) * layers * sizeof(float) * 4 +
                       size_t(vocab) * layers * sizeof(float) + 16 * 1024 * 1024;
        arena.resize(bytes);
        ctx.reset(ggml_init({bytes, arena.data(), false}));
        if (!ctx) throw std::runtime_error("Failed to allocate head graph");
        auto copy = [&](ggml_tensor * src) {
            ggml_tensor * dst = ggml_dup_tensor(ctx.get(), src);
            ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
            return dst;
        };
        auto * weight = copy(model.output);
        auto * norm = copy(model.output_norm);
        input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, layers);
        auto * normalized = ggml_rms_norm(ctx.get(), input, model.hparams.f_norm_rms_eps);
        normalized = ggml_mul(ctx.get(), normalized, norm);
        logits = ggml_mul_mat(ctx.get(), weight, normalized);
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, logits);
        plan = ggml_graph_plan(graph, threads, nullptr);
        work.resize(plan.work_size);
        plan.work_data = work.data();
    }

    const float * run(const capture & c) {
        std::memcpy(input->data, c.states.data(), c.states.size() * sizeof(float));
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Head graph failed");
        }
        return static_cast<const float *>(logits->data);
    }
};

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(128);
    int n = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    }
    if (n < 0) throw std::runtime_error("Token decoding failed");
    return std::string(buf.data(), n);
}

static std::string format_prompt(const llama_model * model, const options & o) {
    if (o.raw) return o.prompt;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("No chat template; provide a formatted --prompt with --raw-prompt");
    llama_chat_message msg{"user", o.prompt.c_str()};
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (n < 0) throw std::runtime_error("Unsupported chat template; use --raw-prompt");
    std::vector<char> buf(n + 1);
    n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
    if (n < 0 || size_t(n) >= buf.size()) throw std::runtime_error("Chat template formatting failed");
    return std::string(buf.data(), n);
}

static int run(const options & o) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = o.gpu_layers;
    mp.use_extra_bufts = false; // Keep head weights in a copyable, standard GGML layout.
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("Model load failed");
    if (model->arch != LLM_ARCH_LLAMA && model->arch != LLM_ARCH_QWEN2 && model->arch != LLM_ARCH_QWEN3) {
        throw std::runtime_error("Unsupported architecture: initial adapters are llama (including Mistral), qwen2, qwen3");
    }
    if (!model->output || !model->output_norm || model->output_b || model->output_norm_b || model->output_s || model->output_in_s) {
        throw std::runtime_error("Expected unscaled linear output head and weighted RMSNorm without bias");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    int width = llama_model_n_embd(model.get()), layers = llama_model_n_layer(model.get());
    int nv = llama_vocab_n_tokens(vocab);
    if (model->output->ne[0] != width || model->output->ne[1] != nv ||
        ggml_nelements(model->output_norm) != width) {
        throw std::runtime_error("Output head dimensions do not match residual stream");
    }
    std::string prompt = format_prompt(model.get(), o);
    int nt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (nt <= 0) throw std::runtime_error("Empty or invalid tokenized prompt");
    std::vector<llama_token> tokens(nt);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), nt, true, true) != nt) {
        throw std::runtime_error("Tokenization failed");
    }
    if (int64_t(nt) + o.tokens > llama_model_n_ctx_train(model.get())) {
        throw std::runtime_error("Prompt + generation exceeds training context; shorten the request");
    }
    capture captured(width, layers);
    auto cp = llama_context_default_params();
    cp.n_ctx = nt + o.tokens;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = o.threads;
    cp.cb_eval = capture::callback;
    cp.cb_eval_user_data = &captured;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("Context creation failed");
    head projection(*model, width, layers, nv, o.threads);

    // Prefill all but the last prompt token without copying intermediate states.
    for (int start = 0; start < nt - 1; start += 512) {
        auto batch = llama_batch_get_one(tokens.data() + start, std::min(512, nt - 1 - start));
        if (llama_decode(ctx.get(), batch)) throw std::runtime_error("Prompt prefill failed");
    }
    auto outpath = std::filesystem::path(o.output);
    if (outpath.has_parent_path()) std::filesystem::create_directories(outpath.parent_path());
    std::ofstream hidden;
    if (o.save_hidden) {
        hidden.open(o.output + ".hidden.f32", std::ios::binary);
        if (!hidden) throw std::runtime_error("Cannot open hidden-state output");
    }
    json steps = json::array();
    std::vector<double> sums(layers, 0);
    std::vector<int> agreements(layers, 0);
    std::vector<std::string> guesses(layers);
    std::string answer;
    double baseline_sum = 0, max_nll_error = 0;
    bool ended = false;
    llama_token current = tokens.back();
    captured.active = true;
    for (int step = 0; step < o.tokens; ++step) {
        captured.reset();
        auto batch = llama_batch_get_one(&current, 1);
        if (llama_decode(ctx.get(), batch)) throw std::runtime_error("Decode failed");
        captured.check();
        const float * native = llama_get_logits_ith(ctx.get(), -1);
        if (!native) throw std::runtime_error("No native logits");
        int target = argmax(native, nv);
        double baseline_nll = token_nll(native, nv, target);
        const float * projected = projection.run(captured);
        double error = std::abs(token_nll(projected + size_t(layers - 1) * nv, nv, target) - baseline_nll);
        max_nll_error = std::max(max_nll_error, error);
        if (error > 0.1) throw std::runtime_error("Final-layer NLL differs from native by >0.1 nat; refusing invalid results");
        bool eog = llama_vocab_is_eog(vocab, target);
        std::string text = piece(vocab, target);
        json nlls = json::array(), predictions = json::array();
        for (int layer = 0; layer < layers; ++layer) {
            const float * row = projected + size_t(layer) * nv;
            double nll = token_nll(row, nv, target);
            int predicted = argmax(row, nv);
            sums[layer] += nll;
            agreements[layer] += predicted == target;
            guesses[layer] += piece(vocab, predicted);
            nlls.push_back(nll);
            predictions.push_back(predicted);
        }
        baseline_sum += baseline_nll;
        steps.push_back({{"step", step}, {"context_position", nt - 1 + step},
                         {"target_token", target}, {"target_piece", text}, {"is_eog", eog},
                         {"native_nll", baseline_nll}, {"layer_nll", nlls}, {"layer_argmax", predictions}});
        if (o.save_hidden) {
            hidden.write(reinterpret_cast<const char *>(captured.states.data()), captured.states.size() * sizeof(float));
            if (!hidden) throw std::runtime_error("Hidden-state write failed");
        }
        if (eog) { ended = true; break; }
        answer += text;
        std::cout << text << std::flush;
        current = target;
    }
    std::cout << "\n";
    const int count = steps.size();
    json summary = json::array();
    std::cout << "\nLayers kept / cut / perplexity / top-1 agreement\n";
    for (int layer = 0; layer < layers; ++layer) {
        double nll = sums[layer] / count;
        double ppl = std::exp(nll);
        summary.push_back({{"layers_kept", layer + 1}, {"layers_cut", layers - layer - 1},
                           {"mean_nll", nll}, {"perplexity", ppl},
                           {"top1_agreement", double(agreements[layer]) / count},
                           {"teacher_forced_argmax_text", guesses[layer]}});
        std::cout << layer + 1 << " / " << layers - layer - 1 << " / " << ppl
                  << " / " << double(agreements[layer]) / count << "\n";
    }
    char arch[128]{};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
    json result = {{"schema_version", 1}, {"llama_revision", LENS_LLAMA_REVISION},
                   {"model_path", std::filesystem::absolute(o.model).string()}, {"architecture", arch},
                   {"head_type", ggml_type_name(model->output->type)},
                   {"normalization", "final weighted RMSNorm"}, {"rms_epsilon", model->hparams.f_norm_rms_eps},
                   {"prompt", o.prompt}, {"formatted_prompt", prompt}, {"prompt_token_ids", tokens},
                   {"sampling", "greedy full-model"}, {"threads", o.threads}, {"gpu_layers", o.gpu_layers},
                   {"requested_tokens", o.tokens}, {"scored_tokens", count}, {"eog_included_in_score", ended},
                   {"answer", answer}, {"native_perplexity", std::exp(baseline_sum / count)},
                   {"max_final_layer_nll_error", max_nll_error}, {"layers", summary}, {"steps", steps}};
    if (o.save_hidden) {
        result["hidden_states"] = {{"path", o.output + ".hidden.f32"},
                                   {"dtype", "float32, native byte order"}, {"shape", {count, layers, width}},
                                   {"layout", "C order: step, layer, hidden; pre-final-norm residual"}};
    }
    std::ofstream output(o.output);
    output << result.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
    output.close();
    if (!output) throw std::runtime_error("Result write failed");
    std::cout << "Saved " << o.output << "\n";
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) { usage(); return 0; }
    try {
        options o = parse(argc, argv);
        llama_backend_init();
        int status = run(o);
        llama_backend_free();
        return status;
    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
