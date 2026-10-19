#pragma once

#include "llama-model.h"
#include "ggml-cpu.h"
#include "residual_capture.h"

#include <cstring>
#include <memory>
#include <vector>

// Reuse GGML's quantized matmul; no dequantization or full FP32 head copy.
class output_head {
public:
    output_head(const llama_model & model, residual_layout layout, int vocabulary_size, int threads) {
        // Storage for copied weights, intermediate activations, logits, and graph metadata.
        constexpr size_t graph_overhead_bytes = 16 * 1024 * 1024;
        // Conservative workspace budget, not a count of actual graph nodes:
        // input + stream additions, plus views/norm/weighted-norm and alignment slack.
        constexpr size_t activation_scratch_slots = 6;
        const size_t weight_bytes = ggml_nbytes(model.output) + ggml_nbytes(model.output_norm);
        const size_t activation_bytes = size_t(layout.width) * layout.layers * sizeof(float) *
                                        (layout.streams * 2 + activation_scratch_slots);
        const size_t logit_bytes = size_t(vocabulary_size) * layout.layers * sizeof(float);
        const size_t arena_bytes = weight_bytes + activation_bytes + logit_bytes + graph_overhead_bytes;
        arena.resize(arena_bytes);
        context.reset(ggml_init({arena_bytes, arena.data(), false}));
        if (!context) {
            throw std::runtime_error("Failed to allocate head graph");
        }

        auto * weight = copy_tensor(model.output);
        auto * norm = copy_tensor(model.output_norm);
        input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, layout.width, layout.streams, layout.layers);
        auto * residual = mean_residual_streams(layout);
        auto * normalized = ggml_rms_norm(context.get(), residual, model.hparams.f_norm_rms_eps);
        normalized = ggml_mul(context.get(), normalized, norm);
        logits = ggml_mul_mat(context.get(), weight, normalized);
        graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, logits);
        plan = ggml_graph_plan(graph, threads, nullptr);
        work.resize(plan.work_size);
        plan.work_data = work.data();
    }

    const float * run(const residual_capture & capture) {
        std::memcpy(input->data, capture.states.data(), capture.states.size() * sizeof(float));
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Head graph failed");
        }
        return static_cast<const float *>(logits->data);
    }

private:
    std::vector<uint8_t> arena, work;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};

    ggml_tensor * copy_tensor(const ggml_tensor * source) {
        auto * destination = ggml_dup_tensor(context.get(), source);
        ggml_backend_tensor_get(source, destination->data, 0, ggml_nbytes(source));
        return destination;
    }

    ggml_tensor * mean_residual_streams(residual_layout layout) {
        // GLM5-Next's native head averages its streams before final RMSNorm.
        // Each view selects one stream across all layers; ordinary models use one view.
        auto * residual = ggml_view_2d(context.get(), input, layout.width, layout.layers, input->nb[2], 0);
        for (int stream = 1; stream < layout.streams; ++stream) {
            auto * view = ggml_view_2d(context.get(), input, layout.width, layout.layers,
                                       input->nb[2], stream * input->nb[1]);
            residual = ggml_add(context.get(), residual, view);
        }
        if (layout.streams > 1) {
            residual = ggml_scale(context.get(), residual, 1.0f / layout.streams);
        }
        return residual;
    }
};
