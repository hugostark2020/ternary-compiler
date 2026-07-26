#pragma once

#include <torch/extension.h>

torch::Tensor ternary_matmul_cpu(
    const torch::Tensor& activations,   // (B, M, K) float32
    const torch::Tensor& packed_weights // (N, K) packed int8
);