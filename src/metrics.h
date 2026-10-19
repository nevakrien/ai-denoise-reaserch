#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

inline int argmax(const float * logits, int n) {
    return int(std::max_element(logits, logits + n) - logits);
}

inline double token_nll(const float * logits, int n, int target) {
    if (n <= 0 || target < 0 || target >= n) {
        throw std::runtime_error("Invalid logit shape or target");
    }
    double peak = logits[argmax(logits, n)];
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("Non-finite logits");
        }
        sum += std::exp(double(logits[i]) - peak);
    }
    return peak + std::log(sum) - logits[target];
}
