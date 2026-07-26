#include "../include/ternary_kernel.h"
#include <cstring>
#include <immintrin.h>


#ifdef __AVX512F__

torch::Tensor ternary_matmul_avx512(const torch::Tensor &activations,
                                    const torch::Tensor &packed_weights) {
  int B = activations.size(0);
  int M = activations.size(1);
  int K = activations.size(2);
  int N = packed_weights.size(0);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor output = torch::empty({B, M, N}, options);

  float *act_ptr = activations.data_ptr<float>();
  int8_t *weight_ptr = packed_weights.data_ptr<int8_t>();
  float *out_ptr = output.data_ptr<float>();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      float *act_row = act_ptr + (b * M + m) * K;
      float *out_row = out_ptr + (b * M + m) * N;

      for (int n = 0; n < N; ++n) {
        __m512 sum_vec = _mm512_setzero_ps();
        int k = 0;
        for (; k + 15 < K; k += 16) {
          __m512 act_vec = _mm512_loadu_ps(act_row + k);
          float weight_vals[16];
          for (int i = 0; i < 16; ++i) {
            int byte_idx = (n * K + k + i) / 4;
            int bit_shift = ((n * K + k + i) % 4) * 2;
            int8_t packed_byte = weight_ptr[byte_idx];
            int w = (packed_byte >> bit_shift) & 0x3;
            weight_vals[i] = (w == 0) ? -1.0f : (w == 1 ? 0.0f : 1.0f);
          }
          __m512 weight_vec = _mm512_loadu_ps(weight_vals);
          sum_vec = _mm512_fmadd_ps(act_vec, weight_vec, sum_vec);
        }
        float sum = _mm512_reduce_add_ps(sum_vec);
        for (; k < K; ++k) {
          int byte_idx = (n * K + k) / 4;
          int bit_shift = ((n * K + k) % 4) * 2;
          int8_t packed_byte = weight_ptr[byte_idx];
          int w = (packed_byte >> bit_shift) & 0x3;
          float weight_val = (w == 0) ? -1.0f : (w == 1 ? 0.0f : 1.0f);
          sum += act_row[k] * weight_val;
        }
        out_row[n] = sum;
      }
    }
  }
  return output;
}

#endif

torch::Tensor ternary_matmul_cpu(const torch::Tensor &activations,
                                 const torch::Tensor &packed_weights) {
#ifdef __AVX512F__
  return ternary_matmul_avx512(activations, packed_weights);
#else
  // Fallback: naive implementation
  int B = activations.size(0);
  int M = activations.size(1);
  int K = activations.size(2);
  int N = packed_weights.size(0);

  auto output = torch::zeros({B, M, N}, torch::kFloat32);
  float *act_ptr = activations.data_ptr<float>();
  int8_t *w_ptr = packed_weights.data_ptr<int8_t>();
  float *out_ptr = output.data_ptr<float>();

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          int byte_idx = (n * K + k) / 4;
          int bit_shift = ((n * K + k) % 4) * 2;
          int w = (w_ptr[byte_idx] >> bit_shift) & 0x3;
          float wf = (w == 0) ? -1.0f : (w == 1 ? 0.0f : 1.0f);
          sum += act_ptr[(b * M + m) * K + k] * wf;
        }
        out_ptr[(b * M + m) * N + n] = sum;
      }
    }
  }
  return output;
#endif
}