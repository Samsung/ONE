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

#include "../KernelGenerator.h"
#include "../Validator.h"

#include "cker/operation/BatchMatMul.h"
#include "cker/operation/FullyConnected.h"
#include "cker/operation/RoPE.h"
#include "cker/operation/SoftMax.h"
#include "cker/operation/Transpose.h"
#include "cker/Shape.h"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Attention &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Attention &node)
{
  using ir::operation::Attention;

  const auto input_index{node.getInputs().at(Attention::Input::INPUT)};
  const auto wq_index{node.getInputs().at(Attention::Input::WQ)};
  const auto wk_index{node.getInputs().at(Attention::Input::WK)};
  const auto wv_index{node.getInputs().at(Attention::Input::WV)};
  const auto wo_index{node.getInputs().at(Attention::Input::WO)};
  const auto cos_index = node.getInputs().at(Attention::Input::COS);
  const auto sin_index = node.getInputs().at(Attention::Input::SIN);
  const auto mask_index = node.getInputs().at(Attention::Input::MASK);
  const auto k_cache_index = node.getInputs().at(Attention::Input::K_CACHE);
  const auto v_cache_index = node.getInputs().at(Attention::Input::V_CACHE);
  const auto pos_index = node.getInputs().at(Attention::Input::POS);

  const auto output_index{node.getOutputs().at(0)};
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto wq_tensor = _tensor_reg->getPortableTensor(wq_index);
  auto wk_tensor = _tensor_reg->getPortableTensor(wk_index);
  auto wv_tensor = _tensor_reg->getPortableTensor(wv_index);
  auto wo_tensor = _tensor_reg->getPortableTensor(wo_index);
  auto cos_tensor = cos_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(cos_index);
  auto sin_tensor = sin_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(sin_index);
  auto mask_tensor = mask_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(mask_index);
  auto k_cache_tensor =
    k_cache_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(k_cache_index);
  auto v_cache_tensor =
    v_cache_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(v_cache_index);
  auto pos_tensor = pos_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(pos_index);

  auto fn = std::make_unique<ops::AttentionLayer>();

  fn->configure(input_tensor, wq_tensor, wk_tensor, wv_tensor, wo_tensor, cos_tensor, sin_tensor,
                mask_tensor, k_cache_tensor, v_cache_tensor, pos_tensor, output_tensor);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

AttentionLayer::AttentionLayer()
  : _input(nullptr), _wq(nullptr), _wk(nullptr), _wv(nullptr), _wo(nullptr), _cos(nullptr),
    _sin(nullptr), _mask(nullptr), _k_cache(nullptr), _v_cache(nullptr), _cache_pos(nullptr),
    _output(nullptr)
{
  // DO NOTHING
}

AttentionLayer::~AttentionLayer() = default;

void AttentionLayer::configure(const IPortableTensor *input, const IPortableTensor *wq,
                               const IPortableTensor *wk, const IPortableTensor *wv,
                               const IPortableTensor *wo, const IPortableTensor *cos,
                               const IPortableTensor *sin, const IPortableTensor *mask,
                               IPortableTensor *k_cache, IPortableTensor *v_cache,
                               const IPortableTensor *pos, IPortableTensor *output)
{
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
  _cache_pos = pos;
  _output = output;

  // 0. Read and check inputs and params
  const auto n_batch = getShape(_input).Dims(0);
  assert(n_batch == 1); // Multi-batch is not supported.
  const auto d_model = getShape(_input).Dims(2);

  if (_cos == nullptr || _sin == nullptr || _cache_pos == nullptr)
    throw std::runtime_error{"Attention: input tensors cannot be nullptr"};

  const auto k_cache_shape = getShape(_k_cache);
  if (k_cache_shape.DimensionsCount() != 4)
    throw std::runtime_error{"K cache tensor must be 4D"};

  // 0.1 Param - Read n_head from K cache 3rd dimension
  const int32_t n_head = k_cache_shape.Dims(2);
  if (d_model % n_head != 0)
    throw std::runtime_error{"d_model must be divisible by n_head"};

  const int32_t d_head = d_model / n_head;
  const auto k_cache_dims = k_cache_shape.DimsData();
  const int32_t k_cache_n_batch = k_cache_dims[0];
  const int32_t k_cache_n_head = k_cache_dims[2];
  const int32_t k_cache_d_head = k_cache_dims[3];

  if (n_batch != k_cache_n_batch || n_head != k_cache_n_head || d_head != k_cache_d_head)
    throw std::runtime_error{"Attention: shape mismatch between inputs"};
}

/**
 * Calculate the block-aligned size that includes the given index.
 *
 * @param index The current token position (0-based index)
 * @param block_size The minimum memory access unit (default: 32)
 * @return The smallest multiple of block_size that can include the given index
 *
 * This function calculates the minimum block-aligned memory size needed to access
 * data from position 0 up to and including the specified index.
 *
 * Example with block_size = 32:
 * - If index = 0 (1st token) -> returns 32 (positions 0-31)
 * - If index = 31 (32nd token) -> returns 32 (positions 0-31)
 *
 * This ensures block-aligned memory access for optimal performance.
 */
int32_t blockSizeFor(int32_t index, int32_t block_size = 32)
{
  // We need to include index, so we calculate for index + 1 elements
  // Then round up to the nearest multiple of block_size
  const int32_t elements_needed = index + 1;
  return ((elements_needed + block_size - 1) / block_size) * block_size;
}

void AttentionLayer::attentionFloat32()
{
  // 0. Read and check inputs and params

  const auto n_batch = getShape(_input).Dims(0);
  assert(n_batch == 1);                           // Multi-batch is not supported.
  const auto n_tokens = getShape(_input).Dims(1); // tokens to decode. Expects 1
  const auto d_model = getShape(_input).Dims(2);

  const int32_t n_head = getShape(_k_cache).Dims(2);
  const int32_t d_head = d_model / n_head;

  // 1. Q, K Projection

  // Input tensor: _input
  //   Shape: [n_batch, n_tokens, d_model]
  //   Data: float*
  // Weight tensors: _wq, _wk, _wv, _wo
  //   Shape: [d_model, d_model] (assuming d_q = d_k = d_v = d_model)
  //   Data: float*

  // Define the output shape for Q and K projections
  nnfw::cker::Shape proj_output_shape({n_batch, n_tokens, d_model});
  nnfw::cker::FullyConnectedParams fc_params{};
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  fc_params.lhs_cacheable = true;

  // Q Projection
  std::vector<float> q_proj_buf(n_batch * n_tokens * d_model);
  nnfw::cker::FullyConnected(fc_params, getShape(_input), getBuffer<float>(_input), getShape(_wq),
                             getBuffer<float>(_wq), /*bias_shape=*/getShape(nullptr),
                             /*bias_data=*/nullptr, proj_output_shape, q_proj_buf.data());

  // K Projection
  std::vector<float> k_proj_buf(n_batch * n_tokens * d_model);
  nnfw::cker::FullyConnected(fc_params, getShape(_input), getBuffer<float>(_input), getShape(_wk),
                             getBuffer<float>(_wk), /*bias_shape=*/getShape(nullptr),
                             /*bias_data=*/nullptr, proj_output_shape, k_proj_buf.data());

  // V Projection
  std::vector<float> v_proj_buf(n_batch * n_tokens * d_model);
  nnfw::cker::FullyConnected(fc_params, getShape(_input), getBuffer<float>(_input), getShape(_wv),
                             getBuffer<float>(_wv), /*bias_shape=*/getShape(nullptr),
                             /*bias_data=*/nullptr, proj_output_shape, v_proj_buf.data());

  // 2. RoPE

  // _cos, _sin
  //   Shape: [ n_batch, n_tokens, d_head ] (assume n_batch=1, n_tokens=1)
  //   DataType: float*

  // 2.1 nullcheck (Validation moved to configure())

  // 2.2  inputs: (input, cos, sin)
  // RoPE expects 4D tensor for input and sin/cos tables.
  // The projection output (proj_out) is logically reinterpreted:
  //     from [ n_batch, n_tokens, d_model ]
  //     to   [ n_batch, n_head, n_tokens, d_head ]
  // This reinterpretation is valid as d_model = n_head * d_head and n_tokens = 1.
  nnfw::cker::Shape rope_in_shape({n_batch, n_head, n_tokens, d_head});
  // Extend cos, sin to 4D
  nnfw::cker::Shape sin_cos_shape({1, n_batch, n_tokens, d_head});

  // 2.3 output buffer:
  nnfw::cker::Shape rope_out_shape({n_batch, n_head, n_tokens, d_head});
  std::vector<float> q_rope_buf(rope_out_shape.FlatSize());
  std::vector<float> k_rope_buf(rope_out_shape.FlatSize());

  nnfw::cker::RoPEMode rope_mode = nnfw::cker::RoPEMode::kGptNeox;

  // 2.4 Call cker::RoPE
  nnfw::cker::RoPE<float>(rope_mode, rope_in_shape, k_proj_buf.data(), sin_cos_shape,
                          getBuffer<float>(_sin), sin_cos_shape, getBuffer<float>(_cos),
                          rope_out_shape, k_rope_buf.data());
  nnfw::cker::RoPE<float>(rope_mode, rope_in_shape, q_proj_buf.data(), sin_cos_shape,
                          getBuffer<float>(_sin), sin_cos_shape, getBuffer<float>(_cos),
                          rope_out_shape, q_rope_buf.data());

  // 3. K cache

  // _k_cache
  //   Shape: [ n_batch, cache_size, n_head, d_head ]
  //   DataType: float

  // 3.1 Transpose K tensor to match k_cache memory layout
  //
  //   from  [ n_batch, n_head, n_tokens, d_head ]
  //     to  [ n_batch, n_tokens, n_head, d_head ]
  //
  // Since both n_tokens and n_batch are 1, the K tensor is already effectively
  // a contiguous memory block of size n_head * d_head, making an explicit transpose
  // operation unnecessary.

  // 3.2 Put K in k_cache[cache_pos]
  const auto k_cache_shape = getShape(_k_cache);
  // _cache_pos is expected to be a 1D tensor with one element, the current sequence position.
  auto cache_pos = static_cast<int64_t>(getBuffer<int64_t>(_cache_pos)[0]);
  if (cache_pos < 0 || cache_pos >= k_cache_shape.Dims(1))
    throw std::runtime_error{"Attention: Current position is out of cache bounds"};

  auto n_k = k_rope_buf.size();
  auto sz_k = sizeof(k_rope_buf[0]) * n_k;
  memcpy(getBuffer<float>(_k_cache) + cache_pos * n_k, k_rope_buf.data(), sz_k);

  // 4. Attension Score

  // 4.1 Transpose K
  const int32_t block_size = 32;
  auto alignedSize = [](int32_t index) -> int32_t {
    // We need to include index, so we calculate for index + 1 elements
    // Then round up to the nearest multiple of block_size
    const int32_t elements_needed = index + 1;
    return ((elements_needed + block_size - 1) / block_size) * block_size;
  };

  const int32_t aligned_ctx_size = alignedSize(cache_pos + 1);
  nnfw::cker::Shape k_cache_tr_in_shape({n_batch, aligned_ctx_size, n_head, d_head});
  nnfw::cker::Shape k_cache_tr_out_shape({n_batch, n_head, aligned_ctx_size, d_head});
  std::vector<float> k_cache_tr_buf(k_cache_tr_out_shape.FlatSize());
  nnfw::cker::TransposeParams tr_params{4, {0, 2, 1, 3}};
  nnfw::cker::Transpose<float>(tr_params, k_cache_tr_in_shape, getBuffer<float>(_k_cache),
                               k_cache_tr_out_shape, k_cache_tr_buf.data());

  // 4.2 Compute attention logits via batch matrix multiplication
  //   Inputs:
  //              q = [n_batch, n_head, n_tokens, d_head]
  // (transposed) k = [n_batch, n_head, aligned_ctx_size, d_head]
  //   Output:
  //             qk = [n_batch, n_head, n_tokens, aligned_ctx_size]
  //
  // Note: Use adj_y=true to avoid additional transpose inside BMM kernel.
  // With adj_y=true, the kernel directly computes Q × K^T without extra
  // memory operations, improving performance by eliminating redundant
  // transpose steps that would otherwise occur internally.
  nnfw::cker::Shape qk_shape({n_batch, n_head, n_tokens, aligned_ctx_size});
  std::vector<float> qk_buf(qk_shape.FlatSize());
  nnfw::cker::BatchMatMul bmm_op;
  bool adj_x = false;
  bool adj_y = true;
  bmm_op.prepare(rope_out_shape, k_cache_tr_out_shape, adj_x, adj_y,
                 /*rhs_const=*/false);
  bmm_op(rope_out_shape, q_rope_buf.data(), k_cache_tr_out_shape, k_cache_tr_buf.data(), adj_x,
         adj_y, qk_shape, qk_buf.data());

  // 4.3 Scale attention logits by 1/sqrt(d_model)
  const float scaling_factor = 1.0f / std::sqrt(static_cast<float>(d_head));
  // NOTE: TICO believes in duality: each tensor seeks its own sqrt(scaling_factor)
  // truth, manifesting as (sqrt(scaling_factor) × Q) * (sqrt(scaling_factor) × K).
  // We embrace unity: one scaling_factor for all, scaling_factor × (Q * K)
  for (size_t i = 0; i < qk_buf.size(); ++i)
    qk_buf[i] *= scaling_factor;

  // 4.4 Add attention

  // mask tensor
  //
  // - shape: [n_batch, n_head, n_tokens, alignged_ctx_sz]
  //   - n_batch  = 1
  //   - n_head   = 1 : mask vector (last axis) is shared across heads
  //   - n_tokens = 1 : Process 1 token per 1 decode
  //   - aligned_ctx_size : multiple of block_size
  // - type: float32
  // - data:
  //      index   0  1  ...     cache_pos    ...           {block_size-1}
  //      mask    0  0 ... 0        0        -inf -inf ...    -inf
  //              |        |                   |                |
  //              +--past--+       new         +-----future-----+
  //                tokens        token              tokens

  const float *mask_data = getBuffer<float>(_mask);
  for (size_t i = 0; i < qk_buf.size(); ++i)
  {
    // The mask repeats every 'aligned_ctx_size' elements.
    qk_buf[i] += mask_data[i % aligned_ctx_size];
  }

  // 4.5 Softmax
  std::vector<float> attn_score_buf(qk_shape.FlatSize());
  nnfw::cker::SoftmaxParams softmax_params;
  softmax_params.beta = 1.0f; // Standard beta for attention softmax
  nnfw::cker::Softmax(softmax_params, qk_shape, qk_buf.data(), qk_shape, attn_score_buf.data());

  // 5. V cache

  // _v_cache
  //   Shape: [n_batch, cache_size, n_head, d_head]
  //   DataType: float

  // 5.1 Transpose V to align v_cache (NOP)
  // 5.2 Put V in v_cache[cache_pos]
  const auto v_cache_shape = getShape(_v_cache);
  if (cache_pos >= v_cache_shape.Dims(1))
    throw std::runtime_error{"Attention: Current position is out of cache bounds"};

  auto n_v = v_proj_buf.size();
  auto sz_v = sizeof(v_proj_buf[0]) * n_v;
  memcpy(getBuffer<float>(_v_cache) + cache_pos * n_v, v_proj_buf.data(), sz_v);

  // 7. Attention Output Calculation
  // 7.1. Compute attention output: attn_scores * V_cache -> attn_out
  nnfw::cker::Shape v_cache_in_shape({n_batch, aligned_ctx_size, n_head, d_head});
  nnfw::cker::Shape v_cache_tr_out_shape({n_batch, n_head, aligned_ctx_size, d_head});
  std::vector<float> v_cache_tr_buf(v_cache_tr_out_shape.FlatSize());
  nnfw::cker::TransposeParams v_tr_params{4, {0, 2, 1, 3}};
  nnfw::cker::Transpose<float>(v_tr_params, v_cache_in_shape, getBuffer<float>(_v_cache),
                               v_cache_tr_out_shape, v_cache_tr_buf.data());

  nnfw::cker::Shape attn_out_shape({n_batch, n_head, n_tokens, d_head});
  std::vector<float> attn_out_buf(attn_out_shape.FlatSize());
  nnfw::cker::BatchMatMul attn_bmm_op;
  attn_bmm_op.prepare(qk_shape, v_cache_tr_out_shape, /*adj_x=*/false, /*adj_y=*/false,
                      /*rhs_const=*/false);
  attn_bmm_op(qk_shape, attn_score_buf.data(), v_cache_tr_out_shape, v_cache_tr_buf.data(),
              /*adj_x=*/false, /*adj_y=*/false, attn_out_shape, attn_out_buf.data());

  // 7.2. Output Projection:
  // When n_batch = n_tokens = 1, attn_out_buf layout already matches [1,1,d_model]
  nnfw::cker::FullyConnected(fc_params, proj_output_shape, attn_out_buf.data(), getShape(_wo),
                             getBuffer<float>(_wo), /*bias_shape=*/getShape(nullptr),
                             /*bias_data=*/nullptr, getShape(_output), getBuffer<float>(_output));
}

void AttentionLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
    attentionFloat32();
  else
    throw std::runtime_error{"AttentionLayer: unsupported input data type"};
}

} // namespace onert::backend::cpu::ops
