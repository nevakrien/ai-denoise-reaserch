#include "residual_capture.h"
#include "ggml-cpu.h"

#include <iostream>
#include <memory>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
static void require_rejected(Function function) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("Expected rejection");
}

static void test_layer_names() {
    require(residual_layer_index("l_out-0", 3) == 0, "First layer");
    require(residual_layer_index("l_out-2", 3) == 2, "Last layer");
    for (const char * name : {"l_out-", "l_out--1", "l_out-3", "l_out-1x", "l_out-+1",
                              "l_out-999999999999999999", "other-0"}) {
        require_rejected([&] { residual_layer_index(name, 3); });
    }
}

static void test_capture(int streams) {
    constexpr int width = 4;
    constexpr int layers = 2;
    constexpr int token_count = 3;
    const residual_layout layout{width, layers, streams};
    residual_capture capture(layout);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(context != nullptr, "Context allocation");

    ggml_tensor * tensor;
    if (streams == 1) {
        tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, token_count);
    } else {
        tensor = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, width, streams, token_count);
    }
    // Use a real CPU backend buffer so tensor_get follows its normal backend path.
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(context.get(), ggml_backend_cpu_buffer_type());
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_owner(buffer, ggml_backend_buffer_free);
    require(buffer != nullptr, "Tensor buffer allocation");
    std::vector<float> values(layout.floats_per_layer() * token_count);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(i);
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));

    ggml_set_name(tensor, "l_out-0");
    require(!residual_capture::callback(tensor, true, &capture), "Inactive capture ignored");
    capture.active = true;
    require(residual_capture::callback(tensor, true, &capture), "Residual requested");
    require(!capture.seen[0], "Ask phase does not copy");
    residual_capture::callback(tensor, false, &capture);
    require_rejected([&] { capture.check(); }); // Layer 1 is still missing.
    ggml_set_name(tensor, "l_out-1");
    residual_capture::callback(tensor, false, &capture);
    capture.check();
    const auto captured_states = capture.states;
    // GLM flattens its final block output before selecting output tokens.
    auto * flattened = ggml_reshape_2d(context.get(), tensor, width * streams, token_count);
    require(std::string(flattened->name) == "l_out-1 (reshaped)", "GGML derived tensor name");
    require(!residual_capture::callback(flattened, true, &capture), "Derived reshape not requested");
    require(!residual_capture::callback(flattened, false, &capture), "Derived reshape not captured");
    auto * restored = ggml_reshape_3d(context.get(), flattened, width, streams, token_count);
    require(!residual_capture::callback(restored, true, &capture), "Nested reshape not requested");
    require(!residual_capture::callback(restored, false, &capture), "Nested reshape not captured");
    capture.check();
    require(capture.states == captured_states, "Derived views leave captured states unchanged");
    const size_t last_token = (token_count - 1) * layout.floats_per_layer();
    for (int layer = 0; layer < layers; ++layer) {
        for (size_t i = 0; i < layout.floats_per_layer(); ++i) {
            require(capture.states[layer * layout.floats_per_layer() + i] == values[last_token + i],
                    "Only last token copied, preserving stream order");
        }
    }

    ggml_set_name(tensor, "other-0");
    require(!residual_capture::callback(tensor, true, &capture), "Unrelated tensor ignored");
    ggml_set_name(tensor, "l_out-invalid");
    residual_capture::callback(tensor, false, &capture);
    require_rejected([&] { capture.check(); }); // No exception crosses callback boundary.
    capture.reset();
    require(capture.error.empty(), "Reset clears errors");
    require_rejected([&] { capture.check(); }); // Reset clears seen layers.

    // Validate layout rejection independently of copying from a malformed tensor.
    const int token_axis = streams == 1 ? 1 : 2;
    tensor->ne[token_axis] = 0;
    require_rejected([&] { layout.last_token_offset(*tensor); });
    tensor->ne[token_axis] = token_count;
    tensor->type = GGML_TYPE_F16;
    require_rejected([&] { layout.last_token_offset(*tensor); });
    tensor->type = GGML_TYPE_F32;
    tensor->ne[3] = 2;
    require_rejected([&] { layout.last_token_offset(*tensor); });
    tensor->ne[3] = 1;
    const size_t token_stride = tensor->nb[token_axis];
    tensor->nb[token_axis] += sizeof(float);
    require_rejected([&] { layout.last_token_offset(*tensor); });
    tensor->nb[token_axis] = token_stride;
    tensor->ne[0] = width + 1;
    require_rejected([&] { layout.last_token_offset(*tensor); });
    tensor->ne[0] = width;
    if (streams == 1) {
        tensor->ne[2] = 2;
    } else {
        tensor->ne[1] = streams + 1;
    }
    require_rejected([&] { layout.last_token_offset(*tensor); });
}

int main() {
    try {
        test_layer_names();
        test_capture(1);
        test_capture(4);
        return 0;
    } catch (const std::exception & exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
