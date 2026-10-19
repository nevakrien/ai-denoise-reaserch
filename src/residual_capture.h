#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// These names and layouts come from the pinned llama.cpp graph builders.
// Single-stream: [width, token, 1, 1]. Multi-stream: [width, stream, token, 1].
inline constexpr std::string_view residual_tensor_prefix = "l_out-";

inline int residual_layer_index(std::string_view name, int layer_count) {
    if (name.substr(0, residual_tensor_prefix.size()) != residual_tensor_prefix) {
        throw std::runtime_error("Not a residual tensor: " + std::string(name));
    }
    const auto suffix = name.substr(residual_tensor_prefix.size());
    int layer = -1;
    const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
    if (parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size() ||
        layer < 0 || layer >= layer_count) {
        throw std::runtime_error("Invalid residual layer index: " + std::string(name));
    }
    return layer;
}

struct residual_layout {
    int width;
    int layers;
    int streams;

    size_t floats_per_layer() const {
        return size_t(width) * streams;
    }

    size_t float_count() const {
        return floats_per_layer() * layers;
    }

    size_t last_token_offset(const ggml_tensor & tensor) const {
        const int token_axis = streams == 1 ? 1 : 2;
        if (tensor.type != GGML_TYPE_F32 || tensor.ne[0] != width ||
            tensor.ne[token_axis] < 1 || tensor.ne[3] != 1 || !ggml_is_contiguous(&tensor)) {
            throw std::runtime_error("Unexpected hidden tensor: " + std::string(tensor.name));
        }
        if (streams == 1 && tensor.ne[2] != 1) {
            throw std::runtime_error("Expected a single residual stream: " + std::string(tensor.name));
        }
        if (streams > 1 && tensor.ne[1] != streams) {
            throw std::runtime_error("Unexpected residual stream count: " + std::string(tensor.name));
        }
        return size_t(tensor.ne[token_axis] - 1) * tensor.nb[token_axis];
    }
};

struct residual_capture {
    residual_layout layout;
    bool active = false;
    // CPU layout: [layer, stream, width], retaining only the last token.
    std::vector<float> states;
    std::vector<bool> seen;
    std::string error;

    explicit residual_capture(residual_layout dimensions)
        : layout(dimensions), states(layout.float_count()), seen(layout.layers, false) {}

    void reset() {
        std::fill(seen.begin(), seen.end(), false);
        error.clear();
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto & capture = *static_cast<residual_capture *>(data);
        if (!capture.active) {
            return false;
        }
        // Do not let exceptions cross the C callback boundary (including name parsing).
        try {
            const std::string_view name = tensor->name;
            if (name.substr(0, residual_tensor_prefix.size()) != residual_tensor_prefix) {
                return false;
            }
            // GGML views inherit names such as "l_out-44 (reshaped)".
            // They are not block outputs and may have a different tensor layout.
            const size_t view_suffix = name.find(" (");
            if (tensor->view_src && view_suffix != std::string_view::npos) {
                residual_layer_index(name.substr(0, view_suffix), capture.layout.layers);
                return false;
            }
            if (ask) {
                return true;
            }
            const int layer = residual_layer_index(name, capture.layout.layers);
            const size_t offset = capture.layout.last_token_offset(*tensor);
            float * destination = capture.states.data() + size_t(layer) * capture.layout.floats_per_layer();
            ggml_backend_tensor_get(tensor, destination, offset,
                                    capture.layout.floats_per_layer() * sizeof(float));
            capture.seen[layer] = true;
        } catch (const std::exception & exception) {
            capture.error = exception.what();
        }
        return true;
    }

    void check() const {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
            throw std::runtime_error("Missing layer tensors; llama.cpp graph does not match this adapter");
        }
    }
};
