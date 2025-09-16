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
  // Expected tensor shapes based on current assumptions:
  // _input: [batch_size=1, seq_len=1, d_model]
  // _wq, _wk, _wv, _wo: [d_model, d_model]
  // _cos, _sin: [batch_size=1, seq_len=1, d_head]
  // _output: [batch_size=1, seq_len=1, d_model] // Assuming WO projects back to d_model

  // Internal buffer shapes during computation:
  // q_proj_buffer (after FullyConnected): [batch_size=1, seq_len=1, d_model]
  // k_proj_buffer (after FullyConnected): [batch_size=1, seq_len=1, d_model]
  // q_reshaped (before RoPE): [batch_size=1, seq_len=1, num_heads, d_head] // d_model = num_heads * d_head
  // k_reshaped (before RoPE): [batch_size=1, seq_len=1, num_heads, d_head] // d_model = num_heads * d_head

  // TODO: Get shapes for all input and output tensors
  // nnfw::cker::Shape input_shape = getShape(_input);
  // nnfw::cker::Shape wq_shape = getShape(_wq);
  // ... and so on for all relevant tensors
  // nnfw::cker::Shape output_shape = getShape(_output);

  // TODO: Call cker Attention kernel functions directly or implement logic here
  // For example:
  // nnfw::cker::AttentionParams params;
  // ... set up params ...
  // nnfw::cker::Attention(params, input_shape, getBuffer<float>(_input), wq_shape, getBuffer<float>(_wq),
  //                       ... , _layer_idx, output_shape, getBuffer<float>(_output));

  throw std::runtime_error{"AttentionLayer::attentionFloat32() not implemented yet."};
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
