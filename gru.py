# Copyright (c) 2020 smarsufan. All Rights Reserved.

import tensorflow as tf
import numpy as np
import ctypes

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

import time

libgru = ctypes.cdll.LoadLibrary('libgru.so')

class GRU:
  def __call__(self, inputs, state):
    self._activation = math_ops.tanh

    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


def ptrof(arr):
  arr = np.ascontiguousarray(arr)
  return arr.ctypes.data_as(ctypes.c_void_p)


def gru1(input, state, gate_kernel, gate_bias, candidate_kernel, candidate_bias):
  # _, input_size = input.shape
  _, hidden_size = state.shape

  input1 = tf.placeholder(dtype=tf.float32, shape=input.shape)
  state1 = tf.placeholder(dtype=tf.float32, shape=state.shape)

  # cell = tf.nn.rnn_cell.GRUCell(hidden_size)
  cell = GRU()
  cell._gate_kernel = tf.constant(gate_kernel)
  cell._gate_bias = tf.constant(gate_bias)
  cell._candidate_kernel = tf.constant(candidate_kernel)
  cell._candidate_bias = tf.constant(candidate_bias)
  output, _ = cell(input1, state1)

  with tf.Session() as sess:
    output1 = sess.run(output, feed_dict={input1: input, state1: state})

  return output1


def gru2(input, state, gate_kernel, gate_bias, candidate_kernel, candidate_bias):
  batch_size, input_size = input.shape
  _, hidden_size = state.shape

  output2 = np.empty(shape=(batch_size, hidden_size), dtype=np.float32)

  t1 = time.time()
  libgru.gru_cell_fp32(
    ptrof(output2),
    ptrof(input), ptrof(state), ptrof(gate_kernel),
    ptrof(gate_bias), ptrof(candidate_kernel), ptrof(candidate_bias),
    batch_size, input_size, hidden_size)
  t2 = time.time()
  print(t2 - t1)

  return output2


if __name__ == '__main__':
  while True:
    batch_size = 1
    input_size = 256
    hidden_size = 256
    input = np.random.uniform(-1, 1, (batch_size, input_size)).astype(np.float32)
    state = np.random.uniform(-1, 1, (batch_size, hidden_size)).astype(np.float32)

    gate_kernel = np.random.uniform(-1, 1, (input_size + hidden_size, 2 * hidden_size)).astype(np.float32)
    gate_bias = np.random.uniform(-1, 1, (2 * hidden_size, )).astype(np.float32)

    candidate_kernel = np.random.uniform(-1, 1, (input_size + hidden_size, hidden_size)).astype(np.float32)
    candidate_bias = np.random.uniform(-1, 1, (hidden_size, )).astype(np.float32)

    output2 = gru2(input, state, np.transpose(gate_kernel, (1, 0)), gate_bias, np.transpose(candidate_kernel, (1, 0)), candidate_bias)
    # print('output2:', output2)

    output1 = gru1(input, state, gate_kernel, gate_bias, candidate_kernel, candidate_bias)
    # print('output1:', output1)

    dis = np.max(np.abs(output1 - output2))
    print(dis)

    if dis > 1e-4:
      break

