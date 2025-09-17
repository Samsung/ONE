/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AttentionLayer.h"
#include "cker/operation/FullyConnected.h"
#include "cker/operation/RoPE.h" // Added for RoPE kernel
#include "cker/Shape.h"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace onert::backend::cpu::ops
{

AttentionLayer::AttentionLayer()
  : _input(nullptr), _wq(nullptr), _wk(nullptr), _wv(nullptr), _wo(nullptr), _cos(nullptr),
    _sin(nullptr), _mask(nullptr), _k_cache(nullptr), _v_cache(nullptr), _pos(nullptr),
    _output(nullptr), _layer_idx(-1)
{
  // DO NOTHING
}

AttentionLayer::~AttentionLayer() = default;

void AttentionLayer::configure(const IPortableTensor *input, const IPortableTensor *wq,
                               const IPortableTensor *wk, const IPortableTensor *wv,
                               const IPortableTensor *wo, const IPortableTensor *cos,
                               const IPortableTensor *sin, const IPortableTensor *mask,
                               const IPortableTensor *k_cache, const IPortableTensor *v_cache,
                               const IPortableTensor *pos, int layer_idx, IPortableTensor *output)
{
  assert(input != nullptr);
  assert(wq != nullptr);
  assert(wk != nullptr);
  assert(wv != nullptr);
  assert(wo != nullptr);
  // Optional tensors can be nullptr
  // assert(cos != nullptr);
  // assert(sin != nullptr);
  // assert(mask != nullptr);
  // assert(k_cache != nullptr);
  // assert(v_cache != nullptr);
  // assert(pos != nullptr);
  assert(output != nullptr);

  _input = input;
  _wq = wq;
  _wk = wk;
  _wv = wv;
  _wo = wo;
  _cos = cos;
  _sin = sin;
  _mask = mask;
  _k_cache = k_cache;
  _v_cache = v_cache;
  _pos = pos;
  _layer_idx = layer_idx;
  _output = output;
}

void AttentionLayer::attentionFloat32()
{
  // Assuming seq_len = 1 for decode-only attention

  // Input tensor: _input
  //   Shape: [batch_size, seq_len, d_model]
  //   Data: float*
  // Weight tensors: _wq, _wk, _wv, _wo
  //   Shape: [d_model, d_model] (assuming d_q = d_k = d_v = d_model for now)
  //   Data: float*
  // RoPE tensors: _cos, _sin
  //   Shape: [1, seq_len, d_head] (batch_size=1, seq_len=1 for decode)
  //   Data: float*
  // Output tensor: _output
  //   Shape: [batch_size, seq_len, d_model]
  //   Data: float*

  // TODO: Call cker Attention kernel functions directly or implement logic here
  // For example:
  // nnfw::cker::AttentionParams params;
  // ... set up params ...
  // nnfw::cker::Attention(params, input_shape, getBuffer<float>(_input), wq_shape,
  // getBuffer<float>(_wq),
  //                       ... , _layer_idx, output_shape, getBuffer<float>(_output));

  const uint32_t batch_size = getShape(_input).Dims(0);
  const uint32_t seq_len = getShape(_input).Dims(1); // Expected to be 1
  const uint32_t d_model = getShape(_input).Dims(2);

  // TODO: d_model should be configurable or derived.
  // For now, assuming num_heads is a known value, e.g., 32.
  // This needs to be aligned with how the model is defined.
  const uint32_t num_heads = 16; // Example value, make this configurable
  if (d_model % num_heads != 0)
  {
    throw std::runtime_error{"d_model must be divisible by num_heads"};
  }
  const uint32_t d_head = d_model / num_heads;

  // Define the output shape for Q and K projections
  int32_t proj_output_dims_array[3] = {
    static_cast<int32_t>(batch_size), static_cast<int32_t>(seq_len), static_cast<int32_t>(d_model)};
  nnfw::cker::Shape proj_output_shape(3, proj_output_dims_array);

  // 1. Q, K, V Projections (using FullyConnected)
  //    Input: [batch_size, seq_len, d_model]
  //    Weights: WQ: [d_model, d_model], WK: [d_model, d_model], WV: [d_model, d_model]
  //    Output: Q_proj: [batch_size, seq_len, d_model], K_proj: [batch_size, seq_len, d_model],
  //    V_proj: [batch_size, seq_len, d_model]

  // Q Projection
  std::vector<float> q_proj_buffer(batch_size * seq_len * d_model);
  nnfw::cker::FullyConnectedParams fc_params_q;
  nnfw::cker::FullyConnected(fc_params_q, getShape(_input), getBuffer<float>(_input), getShape(_wq),
                             getBuffer<float>(_wq), getShape(nullptr) /*bias_shape=*/,
                             /*bias_data=*/nullptr, proj_output_shape, q_proj_buffer.data());

  // K Projection
  std::vector<float> k_proj_buffer(batch_size * seq_len * d_model);
  nnfw::cker::FullyConnectedParams fc_params_k;
  nnfw::cker::FullyConnected(fc_params_k, getShape(_input), getBuffer<float>(_input), getShape(_wk),
                             getBuffer<float>(_wk), getShape(nullptr) /*bias_shape=*/,
                             /*bias_data=*/nullptr, proj_output_shape, k_proj_buffer.data());

  // 2. Apply RoPE to K

  // 2.1 nullcheck
  if (_cos == nullptr || _sin == nullptr)
  {
    throw std::runtime_error{"RoPE _cos and _sin tensors cannot be nullptr"};
  }

  // 2.2 reshape
  // Rope expects 4D tensor for input and sin/con tables.
  int32_t qk_reshaped_dims_array[4] = {
    static_cast<int32_t>(batch_size), static_cast<int32_t>(seq_len),
    static_cast<int32_t>(num_heads), static_cast<int32_t>(d_head)};
  nnfw::cker::Shape qk_reshaped_shape(4, qk_reshaped_dims_array);
  int32_t sin_cos_dims[4] = {1, 1, 1, static_cast<int32_t>(d_head)};
  nnfw::cker::Shape sin_cos_shape(4, sin_cos_dims);

  // 2.3 prepare output buffer
  std::vector<float> q_rope_buffer(batch_size * seq_len * num_heads * d_head);
  std::vector<float> k_rope_buffer(batch_size * seq_len * num_heads * d_head);

  nnfw::cker::RoPEMode rope_mode = nnfw::cker::RoPEMode::kGptNeox;

  // 2.4 finally call kernel
  nnfw::cker::RoPE<float>(rope_mode, qk_reshaped_shape, k_proj_buffer.data(), sin_cos_shape,
                          getBuffer<float>(_sin), sin_cos_shape, getBuffer<float>(_cos),
                          qk_reshaped_shape, k_rope_buffer.data());
  nnfw::cker::RoPE<float>(rope_mode, qk_reshaped_shape, q_proj_buffer.data(), sin_cos_shape,
                          getBuffer<float>(_sin), sin_cos_shape, getBuffer<float>(_cos),
                          qk_reshaped_shape, q_rope_buffer.data());

  // Next steps: V projection, attention scores, softmax, etc.
}

void AttentionLayer::run()
{
  // Assuming FLOAT32 for now, add other data types later
  if (_input->data_type() == OperandType::FLOAT32)
  {
    // TODO: Add checks for other input tensor data types if they can differ
    attentionFloat32();
  }
  else
  {
    throw std::runtime_error{"AttentionLayer: unsupported input data type"};
  }
}

} // namespace onert::backend::cpu::ops
