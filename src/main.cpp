#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include "metrics.h"
#include "output_head.h"
#include "residual_capture.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string prompt = "Why is the sky blue?";
    std::string output = "results/run.json";
    std::string requests;
    int tokens = 32;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    int gpu_layers = 0;
    bool raw = false;
    bool save_hidden = false;
};

static void usage() {
    std::cout << "gguf-lens --model PATH [--prompt QUESTION] [--tokens 32]\n"
                 "          [--output results/run.json] [--threads N] [--gpu-layers N]\n"
                 "          [--raw-prompt] [--save-hidden] [--requests requests.json]\n"
                 "Full-model greedy generation; every layer scores the same generated tokens.\n";
}

static options parse(int argc, char ** argv) {
    options settings;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--raw-prompt") {
            settings.raw = true;
            continue;
        }
        if (key == "--save-hidden") {
            settings.save_hidden = true;
            continue;
        }
        if (i + 1 == argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        std::string value = argv[++i];
        if (key == "--model") {
            settings.model = value;
        } else if (key == "--prompt") {
            settings.prompt = value;
        } else if (key == "--output") {
            settings.output = value;
        } else if (key == "--tokens") {
            settings.tokens = std::stoi(value);
        } else if (key == "--threads") {
            settings.threads = std::stoi(value);
        } else if (key == "--gpu-layers") {
            settings.gpu_layers = std::stoi(value);
        } else if (key == "--requests") {
            settings.requests = value;
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (settings.model.empty() || settings.prompt.empty() || settings.output.empty() ||
        settings.tokens <= 0 || settings.threads <= 0) {
        throw std::runtime_error("Model, nonempty prompt/output, and positive tokens/threads required");
    }
    return settings;
}

static constexpr int prefill_batch_size = 512;
static constexpr double max_final_layer_nll_difference = 0.1;

static residual_layout model_residual_layout(const llama_model * model) {
    int streams = 1;
    if (model->arch == LLM_ARCH_GLM5_NEXT) {
        streams = model->hparams.dsv4_hc_mult;
    }
    return {llama_model_n_embd(model), llama_model_n_layer(model), streams};
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(128);
    int length = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (length < 0) {
        buffer.resize(-length);
        length = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    if (length < 0) {
        throw std::runtime_error("Token decoding failed");
    }
    return std::string(buffer.data(), length);
}

static std::string format_prompt(const llama_model * model, const options & settings) {
    if (settings.raw) {
        return settings.prompt;
    }
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) {
        throw std::runtime_error("No chat template; provide a formatted --prompt with --raw-prompt");
    }
    llama_chat_message message{"user", settings.prompt.c_str()};
    int length = llama_chat_apply_template(chat_template, &message, 1, true, nullptr, 0);
    if (length < 0) {
        throw std::runtime_error("Unsupported chat template; use --raw-prompt");
    }
    std::vector<char> buffer(length + 1);
    length = llama_chat_apply_template(chat_template, &message, 1, true, buffer.data(), buffer.size());
    if (length < 0 || size_t(length) >= buffer.size()) {
        throw std::runtime_error("Chat template formatting failed");
    }
    return std::string(buffer.data(), length);
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int token_count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (token_count <= 0) {
        throw std::runtime_error("Empty or invalid tokenized prompt");
    }
    std::vector<llama_token> tokens(token_count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), token_count, true, true) != token_count) {
        throw std::runtime_error("Tokenization failed");
    }
    return tokens;
}

static void write_result(const std::string & path, const json & result) {
    const std::string temporary_path = path + ".tmp";
    std::ofstream output(temporary_path);
    output << result.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("Result write failed");
    }
    std::filesystem::rename(temporary_path, path);
    std::cout << "Saved " << path << "\n";
}

static void measure(const options & settings, llama_model * model, output_head & projection) {
    const auto started = std::chrono::steady_clock::now();
    const auto * vocab = llama_model_get_vocab(model);
    const auto layout = model_residual_layout(model);
    const int width = layout.width;
    const int layers = layout.layers;
    const int streams = layout.streams;
    const int vocabulary_size = llama_vocab_n_tokens(vocab);
    const std::string prompt = format_prompt(model, settings);
    auto tokens = tokenize_prompt(vocab, prompt);
    const int prompt_token_count = tokens.size();
    if (int64_t(prompt_token_count) + settings.tokens > llama_model_n_ctx_train(model)) {
        throw std::runtime_error("Prompt + generation exceeds training context; shorten the request");
    }
    residual_capture captured(layout);
    auto context_params = llama_context_default_params();
    context_params.n_ctx = prompt_token_count + settings.tokens;
    context_params.n_batch = prefill_batch_size;
    context_params.n_ubatch = prefill_batch_size;
    context_params.n_threads = settings.threads;
    context_params.n_threads_batch = settings.threads;
    context_params.cb_eval = residual_capture::callback;
    context_params.cb_eval_user_data = &captured;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model, context_params), llama_free);
    if (!context) {
        throw std::runtime_error("Context creation failed");
    }

    // Prefill all but the last prompt token without copying intermediate states.
    for (int start = 0; start < prompt_token_count - 1; start += prefill_batch_size) {
        const int batch_size = std::min(prefill_batch_size, prompt_token_count - 1 - start);
        auto batch = llama_batch_get_one(tokens.data() + start, batch_size);
        if (llama_decode(context.get(), batch)) {
            throw std::runtime_error("Prompt prefill failed");
        }
    }
    const auto output_path = std::filesystem::path(settings.output);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream hidden;
    if (settings.save_hidden) {
        hidden.open(settings.output + ".hidden.f32", std::ios::binary);
        if (!hidden) {
            throw std::runtime_error("Cannot open hidden-state output");
        }
    }
    json steps = json::array();
    std::vector<double> sums(layers, 0);
    std::vector<int> agreements(layers, 0);
    std::vector<std::string> guesses(layers);
    std::string answer;
    double baseline_sum = 0;
    double max_nll_error = 0;
    bool ended = false;
    llama_token current = tokens.back();
    captured.active = true;
    for (int step = 0; step < settings.tokens; ++step) {
        captured.reset();
        auto batch = llama_batch_get_one(&current, 1);
        if (llama_decode(context.get(), batch)) {
            throw std::runtime_error("Decode failed");
        }
        captured.check();
        const float * native = llama_get_logits_ith(context.get(), -1);
        if (!native) {
            throw std::runtime_error("No native logits");
        }
        const int target = argmax(native, vocabulary_size);
        const double baseline_nll = token_nll(native, vocabulary_size, target);
        const float * projected = projection.run(captured);
        const float * final_layer_logits = projected + size_t(layers - 1) * vocabulary_size;
        const double final_layer_nll = token_nll(final_layer_logits, vocabulary_size, target);
        const double error = std::abs(final_layer_nll - baseline_nll);
        max_nll_error = std::max(max_nll_error, error);
        if (error > max_final_layer_nll_difference) {
            throw std::runtime_error("Final-layer NLL differs from native by more than the allowed limit; refusing invalid results");
        }
        const bool eog = llama_vocab_is_eog(vocab, target);
        const std::string text = piece(vocab, target);
        json nlls = json::array();
        json predictions = json::array();
        for (int layer = 0; layer < layers; ++layer) {
            const float * row = projected + size_t(layer) * vocabulary_size;
            const double nll = token_nll(row, vocabulary_size, target);
            const int predicted = argmax(row, vocabulary_size);
            sums[layer] += nll;
            agreements[layer] += predicted == target;
            guesses[layer] += piece(vocab, predicted);
            nlls.push_back(nll);
            predictions.push_back(predicted);
        }
        baseline_sum += baseline_nll;
        steps.push_back({{"step", step}, {"context_position", prompt_token_count - 1 + step},
                         {"target_token", target}, {"target_piece", text}, {"is_eog", eog},
                         {"native_nll", baseline_nll}, {"layer_nll", nlls}, {"layer_argmax", predictions}});
        if (settings.save_hidden) {
            hidden.write(reinterpret_cast<const char *>(captured.states.data()), captured.states.size() * sizeof(float));
            if (!hidden) {
                throw std::runtime_error("Hidden-state write failed");
            }
        }
        if (eog) {
            ended = true;
            break;
        }
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
    char name[512]{};
    char file_type[128]{};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    llama_model_meta_val_str(model, "general.name", name, sizeof(name));
    llama_model_meta_val_str(model, "general.file_type", file_type, sizeof(file_type));
    json result = {{"schema_version", 2}, {"llama_revision", LENS_LLAMA_REVISION},
                   {"model_path", std::filesystem::absolute(settings.model).string()}, {"architecture", arch},
                   {"model_name", name}, {"file_type", file_type}, {"residual_streams", streams},
                   {"native_head_placement", "CPU (including final norm and tied embeddings)"},
                   {"stream_reduction", streams > 1 ? "arithmetic mean before final RMSNorm" : "none"},
                   {"head_type", ggml_type_name(model->output->type)},
                   {"normalization", "final weighted RMSNorm"}, {"rms_epsilon", model->hparams.f_norm_rms_eps},
                   {"prompt", settings.prompt}, {"formatted_prompt", prompt}, {"prompt_token_ids", tokens},
                   {"sampling", "greedy full-model"}, {"threads", settings.threads}, {"gpu_layers", settings.gpu_layers},
                   {"requested_tokens", settings.tokens}, {"scored_tokens", count}, {"eog_included_in_score", ended},
                   {"answer", answer}, {"native_perplexity", std::exp(baseline_sum / count)},
                   {"elapsed_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()},
                   {"max_final_layer_nll_error", max_nll_error}, {"layers", summary}, {"steps", steps}};
    if (settings.save_hidden) {
        json shape = {count, layers};
        if (streams > 1) {
            shape.push_back(streams);
        }
        shape.push_back(width);
        result["hidden_states"] = {{"path", settings.output + ".hidden.f32"},
                                   {"dtype", "float32, native byte order"}, {"shape", shape},
                                   {"layout", "C order: step, layer, [stream,] hidden; pre-final-norm residual"}};
    }
    write_result(settings.output, result);
}

static void validate_model(const llama_model & model) {
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_GLM5_NEXT:
            break;
        default:
            throw std::runtime_error("Unsupported architecture: no validated graph convention for this family");
    }
    const bool missing_weights = !model.output || !model.output_norm;
    const bool has_bias = model.output_b || model.output_norm_b;
    const bool has_scale = model.output_s || model.output_in_s;
    if (missing_weights || has_bias || has_scale) {
        throw std::runtime_error("Expected unscaled linear output head and weighted RMSNorm without bias");
    }
    const auto layout = model_residual_layout(&model);
    const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(&model));
    if (model.output->ne[0] != layout.width || model.output->ne[1] != vocabulary_size ||
        ggml_nelements(model.output_norm) != layout.width) {
        throw std::runtime_error("Output head dimensions do not match residual stream");
    }
}

static int measure_requests(const options & settings, llama_model * model, output_head & projection) {
    std::ifstream input(settings.requests);
    json requests;
    input >> requests;
    if (!requests.is_array() || requests.empty()) {
        throw std::runtime_error("Expected nonempty request array");
    }
    int failures = 0;
    for (const auto & request : requests) {
        options request_settings = settings;
        request_settings.prompt = request.at("prompt").get<std::string>();
        request_settings.output = request.at("output").get<std::string>();
        request_settings.raw = request.value("raw_prompt", settings.raw);
        if (request_settings.prompt.empty() || request_settings.output.empty()) {
            throw std::runtime_error("Empty request prompt/output");
        }
        try {
            measure(request_settings, model, projection);
        } catch (const std::exception & exception) {
            std::cerr << "Request failed: " << request_settings.output << ": " << exception.what() << '\n';
            ++failures;
        }
    }
    if (failures > 0) {
        return 2;
    }
    return 0;
}

static int run(const options & settings) {
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = settings.gpu_layers;
    model_params.use_extra_bufts = false; // Keep head weights in a copyable, standard GGML layout.
    // Match native and reconstructed head arithmetic, even with GPU transformer blocks.
    llama_model_tensor_buft_override head_placement[] = {
        {"^(output|output_norm|token_embd)\\.weight$", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    model_params.tensor_buft_overrides = head_placement;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(settings.model.c_str(), model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("Model load failed");
    }
    validate_model(*model);
    const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    output_head projection(*model, model_residual_layout(model.get()), vocabulary_size, settings.threads);
    if (!settings.requests.empty()) {
        return measure_requests(settings, model.get(), projection);
    }
    measure(settings, model.get(), projection);
    return 0;
}

struct backend_lifetime {
    backend_lifetime() { llama_backend_init(); }
    ~backend_lifetime() { llama_backend_free(); }
    backend_lifetime(const backend_lifetime &) = delete;
    backend_lifetime & operator=(const backend_lifetime &) = delete;
};

int main(int argc, char ** argv) {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        usage();
        return 0;
    }
    try {
        const options settings = parse(argc, argv);
        const backend_lifetime backend;
        return run(settings);
    } catch (const std::exception & exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
