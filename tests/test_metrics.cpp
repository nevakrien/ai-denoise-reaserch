#include "metrics.h"
#include <iostream>

int main() {
    const float uniform[] = {0, 0, 0, 0};
    const float logits[] = {1, 2, 3};
    const float shifted[] = {1001, 1002, 1003};
    if (std::abs(token_nll(uniform, 4, 2) - std::log(4.0)) > 1e-12 ||
        std::abs(token_nll(logits, 3, 1) - token_nll(shifted, 3, 1)) > 1e-12 ||
        argmax(logits, 3) != 2) {
        std::cerr << "Metric test failed\n";
        return 1;
    }
    try {
        token_nll(logits, 3, 3);
        return 1;
    } catch (const std::runtime_error &) {}
    return 0;
}
