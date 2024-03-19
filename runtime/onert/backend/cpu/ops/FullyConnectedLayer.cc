/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FullyConnectedLayer.h"

#include "../Tensor.h"
#include <cker/operation/FullyConnected.h>
#include <cker/TensorUtils.h>
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

FullyConnectedLayer::FullyConnectedLayer()
  : _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr),
    _activation(ir::Activation::NONE), _temp_arena(new nnfw::cker::FCTempArena()),
    _external_context(nullptr), _is_hybrid(false), _is_shuffled16x1float32(false)
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::fullyConnectedFloat32()
{
  nnfw::cker::FullyConnectedParams op_params;
  float output_activation_min = 0;
  float output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  op_params.activation = convertActivationType(_activation);
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  // TODO Set both cachables as false when training
  op_params.lhs_cacheable = _weights->is_constant();
  op_params.rhs_cacheable = _input->is_constant();

  nnfw::cker::FullyConnected(op_params, getShape(_input), getBuffer<float>(_input),
                             getShape(_weights), getBuffer<float>(_weights), getShape(_bias),
                             _bias ? getBuffer<float>(_bias) : nullptr, getShape(_output),
                             getBuffer<float>(_output));
}

// executionMutex is used to protect concurrent access of non-threadsafe resources
// like gemmlowp::GemmContext.
void FullyConnectedLayer::fullyConnectedQuant8()
{
  double real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  GetQuantizedConvolutionMultiplier(_input, _weights, _bias, _output, &real_multiplier);
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.input_offset = -_input->data_zero_point();
  op_params.weights_offset = -_weights->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::FullyConnected(op_params, getShape(_input), getBuffer<uint8_t>(_input),
                             getShape(_weights), getBuffer<uint8_t>(_weights), getShape(_bias),
                             _bias ? getBuffer<int32_t>(_bias) : nullptr, getShape(_output),
                             getBuffer<uint8_t>(_output));
}

void FullyConnectedLayer::fullyConnectedHybrid()
{
  nnfw::cker::FCTempArena &temp_arena = *_temp_arena;
  if (!temp_arena.prepared)
  {
    temp_arena.prepare(getShape(_input), getShape(_weights));
  }

  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);
  op_params.weights_scale = _weights->data_scale();

#ifndef USE_RUY_GEMV
  nnfw::cker::FullyConnectedHybrid(
    op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
    getBuffer<int8_t>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
    getShape(_output), getBuffer<float>(_output), temp_arena, _external_context->ruy_context());
#else
  nnfw::cker::FullyConnectedHybrid(
    op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
    (_cached_weights) ? reinterpret_cast<const int8_t *>(_cached_weights)
                      : getBuffer<int8_t>(_weights),
    getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr, getShape(_output),
    getBuffer<float>(_output), temp_arena, _external_context->ruy_context());

  if (_cached_weights == nullptr || _is_weights_freed)
    return;

  // '_cached_weights is not nullptr and _is_weights_freed is false' means
  // this weight shape is satisfied with the ruy kernel's prepack cache's condition.
  // After entering here, it will not enter again except below the case - input is zero-vector

  // if input's elements are filled with zero, it by-passes(does not enter ruy-kernel path)
  // so that handle this case
  const int input_size = getShape(_input).FlatSize();
  if (nnfw::cker::IsZeroVector(getBuffer<float>(_input), input_size))
    return;

  auto weight_tensor = nnfw::misc::polymorphic_downcast<const Tensor *>(_weights);

  // This weight tensor could be other ops' const tensor.
  // Therefore, below reference should be checked like following
  auto tensor = const_cast<Tensor *>(weight_tensor);
  if (tensor->buffer() == nullptr) // ref is already 0?
  {
    _is_weights_freed = true;
    return;
  }

  tensor->decrease_ref();
  if (tensor->buffer() == nullptr) // ref == 0?
  {
#if defined(__ANDROID__) && (__ANDROID_API__ >= 26)
    // NOTE This line forces OS to release any unused memory immediately
    mallopt(M_PURGE, 0);
#endif
    _is_weights_freed = true;
  }
#endif
}

void FullyConnectedLayer::fullyConnectedSparseWeight()
{
  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);

  const uint16_t *w1_segments = _weights->sparsity()->w1_segments();
  const uint16_t *w1_indices = _weights->sparsity()->w1_indices();

  auto block_size = _weights->sparsity()->block_size();
  if (block_size.size() == 0)
  {
    nnfw::cker::FullyConnectedSparseWeightRandom(
      op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
      getBuffer<float>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
      getShape(_output), getBuffer<float>(_output), w1_segments, w1_indices);
  }
  else if (block_size.size() == 2 && block_size[0] == 16 && block_size[1] == 1)
  {
    nnfw::cker::FullyConnectedSparseWeight16x1(
      op_params, getShape(_input), getBuffer<float>(_input), getShape(_weights),
      getBuffer<float>(_weights), getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
      getShape(_output), getBuffer<float>(_output), w1_segments, w1_indices);
  }
  else
    throw std::runtime_error{"FullyConnected: unsupported sparsity"};
}

void FullyConnectedLayer::fullyConnected16x1Float32()
{
#if defined(__aarch64__) && defined(USE_NEON)
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);

  nnfw::cker::FullyConnected16x1Float32(op_params, getShape(_input), getBuffer<float>(_input),
                                        getShape(_weights), getBuffer<float>(_weights),
                                        getShape(_bias), _bias ? getBuffer<float>(_bias) : nullptr,
                                        getShape(_output), getBuffer<float>(_output));
#else
  throw std::runtime_error{"FullyConnected: Shuffled16x1Float32 weights_format is not supported."};
#endif
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    ir::FullyConnectedWeightsFormat weights_format,
                                    IPortableTensor *output,
                                    const std::shared_ptr<ExternalContext> &external_context)
{
  _input = input;
  _weights = weights;
  _bias = bias;
  _activation = activation;
  _output = output;
  _is_hybrid = input->data_type() == OperandType::FLOAT32 &&
               weights->data_type() == OperandType::QUANT_INT8_SYMM;
  _is_hybrid = input->data_type() == OperandType::FLOAT32 &&
               weights->data_type() == OperandType::QUANT_INT4_SYMM;

  _is_shuffled16x1float32 = weights_format == ir::FullyConnectedWeightsFormat::Shuffled16x1Float32;
#if !defined(__aarch64__) || !defined(USE_NEON)
  if (_is_shuffled16x1float32)
  {
    throw std::runtime_error{
      "FullyConnected: Shuffled16x1Float32 weights_format is not supported."};
  }
#endif
  _external_context = external_context;
}

void FullyConnectedLayer::run()
{
  if (_is_hybrid)
  {
    fullyConnectedHybrid();
  }
  else if (_weights->sparsity())
  {
    fullyConnectedSparseWeight();
  }
  else if (_input->data_type() == OperandType::FLOAT32)
  {
    _is_shuffled16x1float32 ? fullyConnected16x1Float32() : fullyConnectedFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    fullyConnectedQuant8();
  }
  else
  {
    throw std::runtime_error{"FullyConnected: unsupported data type"};
  }
}

void FullyConnectedLayer::prepare()
{
  if (_bias && _bias->is_constant())
  {
    const int bias_size = getShape(_bias).FlatSize();
    if (nnfw::cker::IsZeroVector(getBuffer<float>(_bias), bias_size))
    {
      _bias = nullptr;
    }
  }

#if (defined(__ARM_NEON__) || defined(__ARM_NEON)) && defined(USE_RUY_GEMV)
  // TODO This is workaround
  // The only fc hybrid will use ruy kernel
  if (_input->data_type() != OperandType::FLOAT32 ||
      _weights->data_type() != OperandType::QUANT_INT8_SYMM)
  {
    return;
  }

  // NOTE. The condition to enable caching on ruy kernel can be changed according to ruy's version

  // If input is dynamic, it changes total size of input
  // If weights is not constant, weights cannot be cached
  if (_input->is_dynamic() || !_weights->is_constant())
    return;

  const int rows = getShape(_weights).Dims(0);
  if (rows % 4 == 0)
  {
    // TODO If it's possible to extract precaching from ruy kernel,
    // place this instead of below code

    // buffer will be used by ruy kernel as a cache key
    _cached_weights = _weights->buffer();
  }
#endif
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
