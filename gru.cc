// Copyright (c) 2020 smarsufan. All Rights Reserved.

#include <vector>
#include <cmath>

template <typename T>
void tanh(T *y, T *x, int size) {
  for (int idx = 0; idx < size; ++idx) {
    y[idx] = tanhf(x[idx]);
  }
}

template <typename T>
void sigmoid(T *y, T *x, int size) {
  for (int idx = 0; idx < size; ++idx) {
    y[idx] = 1. / (1. + expf(-x[idx]));
  }
}

template <typename T>
void matmul(T *C, T *A, T *B, T *bias, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T sum = 0;
      for (int c = 0; c < k; ++c) {
        sum += A[i * k + c] * B[n * k + c];
      }
      C[i * n + j] = sum + bias[j];
    }
  }
}

/** Contat [n, c1], [n, c2] to [n, c1 + c2]
 */
template <typename T>
void concate(T *C, T *A, T *B, int n, int c1, int c2) {
  T *p = C;
  for (int i = 0; i < n; ++i) {
    memcpy(p, A + i * c1, sizeof(T) * c1);
    p += c1;
    memcpy(p, B + i * c2, sizeof(T) * c2);
    p += c2;
  }
}

/**
 * GRU Implement.
 * 
 * TODO(smarsufan): Optimizer the use of memory.
 * 
 * @param inputs: shape [batch_size, input_size]. The input is supposed to be [T, N, C] and T in time t.
 * @param state: shape [batch_size, hidden_size]. The hidden state.
 * @param gate_kernel: shape [hidden_size * 2, hidden_size + input_size]
 * @param gate_bias: shape [hidden_size * 2]
 * @param candidate_kernel: shape [hidden_size, hidden_size + input_size]
 * @param candidate_bias: shape [hidden_size]
 * 
 */
template <typename T>
void gru_cell(T *outputs, T *inputs, T *state, T *gate_kernel, T *gate_bias, T *candidate_kernel, T *candidate_bias, int batch_size, int input_size, int hidden_size) {
  /// 1. Concat input and state in dim 1.
  std::vector<T> concat_inputs(batch_size * (input_size + hidden_size));
  concate(concat_inputs.data(), inputs, state, batch_size, input_size, hidden_size);

  /// 2. Compute gate_inputs
  std::vector<T> gate_inputs(batch_size * (hidden_size * 2));
  matmul(gate_inputs.data(), concat_inputs.data(), gate_kernel, gate_bias, batch_size, hidden_size * 2, hidden_size + input_size);

  /// 3. Compute sigmoid value
  std::vector<T> value(batch_size * (hidden_size * 2));
  sigmoid(value.data(), gate_inputs.data(), gate_inputs.size());

  /// 4. Split value to r, u  
  std::vector<T> r(batch_size * hidden_size);
  std::vector<T> u(batch_size * hidden_size);
  T *value_p = value.data();
  for (int i = 0; i < batch_size; ++i) {
    memcpy(r.data() + i * hidden_size, value_p, sizeof(T) * hidden_size);
    value_p += hidden_size;
    memcpy(u.data() + i * hidden_size, value_p, sizeof(T) * hidden_size);
    value_p += hidden_size;
  }

  std::vector<T> r_state(batch_size * hidden_size);
  for (int i = 0; i < r_state.size(); ++i) {
    r_state[i] = r[i] * state[i];
  }

  /// 5. Concat inputs and r_state to candidate_inputs
  std::vector<T> candidate_inputs(batch_size * (input_size + hidden_size));
  concate(candidate_inputs.data(), inputs, r_state.data(), batch_size, input_size, hidden_size);

  /// 6. Compute candidate
  std::vector<T> candidate(batch_size * hidden_size);
  matmul(candidate.data(), candidate_inputs.data(), candidate_kernel, candidate_bias, batch_size, hidden_size, input_size + hidden_size);

  /// 7. tanh actv
  std::vector<T> c(batch_size * hidden_size);
  tanh(c.data(), candidate.data(), c.size());

  /// 8. Compute result
  for (int idx = 0; idx < batch_size * hidden_size; ++idx) {
    outputs[idx] = u[idx] * state[idx] + (1 - u[idx]) * c[idx];
  }
}
