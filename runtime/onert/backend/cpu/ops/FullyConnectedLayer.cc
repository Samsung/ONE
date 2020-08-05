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
      _external_context(nullptr), _is_hybrid(false)
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::fullyConnectedFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  op_params.activation = convertActivationType(_activation);

  nnfw::cker::FullyConnected(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_weights), reinterpret_cast<const float *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
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
  CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                &output_activation_max);

  nnfw::cker::FullyConnectedParams op_params;
  op_params.input_offset = -_input->data_offset();
  op_params.weights_offset = -_weights->data_offset();
  op_params.output_offset = _output->data_offset();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::FullyConnected(
      op_params, getTensorShape(_input), reinterpret_cast<const uint8_t *>(_input->buffer()),
      getTensorShape(_weights), reinterpret_cast<const uint8_t *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const int32_t *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
}

void FullyConnectedLayer::fullyConnectedHybrid()
{
  nnfw::cker::FCTempArena &temp_arena = *_temp_arena;
  if (!temp_arena.prepared)
  {
    temp_arena.prepare(getTensorShape(_input), getTensorShape(_weights));
  }

  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = convertActivationType(_activation);
  op_params.weights_scale = _weights->data_scale();

#ifndef USE_RUY_GEMV
  nnfw::cker::FullyConnectedHybrid(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_weights), reinterpret_cast<const int8_t *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()), temp_arena,
      _external_context->ruy_context());
#else
  nnfw::cker::FullyConnectedHybrid(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_weights),
      (_cached_weights) ? reinterpret_cast<const int8_t *>(_cached_weights)
                        : reinterpret_cast<const int8_t *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()), temp_arena,
      _external_context->ruy_context());

// TODO Remove this ifdef
#ifdef EXPERIMENTAL_RUY_FEATURE
  if (_cached_weights == nullptr || _is_weights_freed)
    return;

  // '_cached_weights is not nullptr and _is_weights_freed is false' means
  // this weight shape is satisfied with the ruy kernel's prepack cache's condition.
  // After entering here, it will not enter again except below the case - input is zero-vector

  // if input's elements are filled with zero, it by-passes(does not enter ruy-kernel path)
  // so that handle this case
  const int input_size = getTensorShape(_input).FlatSize();
  if (nnfw::cker::IsZeroVector(reinterpret_cast<float *>(_input->buffer()), input_size))
    return;

  // This weight tensor could be other ops' const tensor.
  // Therefore, below reference should be checked like following
  auto weight_tensor = dynamic_cast<const Tensor *>(_weights);
  if (weight_tensor)
  {
    auto tensor = const_cast<Tensor *>(weight_tensor);
    if (tensor->buffer() == nullptr) // ref is already 0?
    {
      _is_weights_freed = true;
      return;
    }

    tensor->decrease_ref();
    if (tensor->buffer() == nullptr) // ref == 0?
    {
      _is_weights_freed = true;
    }
  }
#endif
#endif
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
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
  _external_context = external_context;
}

void FullyConnectedLayer::run()
{
  if (_is_hybrid)
  {
    fullyConnectedHybrid();
  }
  else if (_input->data_type() == OperandType::FLOAT32)
  {
    fullyConnectedFloat32();
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
    const int bias_size = getTensorShape(_bias).FlatSize();
    if (nnfw::cker::IsZeroVector(reinterpret_cast<float *>(_bias->buffer()), bias_size))
    {
      _bias = nullptr;
    }
  }

#ifdef USE_RUY_GEMV
#ifdef EXPERIMENTAL_RUY_FEATURE
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

  const int rows = getTensorShape(_weights).Dims(0);
  if (rows % 4 == 0)
  {
    const int total_input_size = getTensorShape(_input).FlatSize();
    const int input_size = getTensorShape(_weights).Dims(1);
    const int batch_size = total_input_size / input_size;
    if (batch_size <= 4)
    {
      // TODO If it's possible to extract precaching from ruy kernel,
      // place this instead of below code

      // buffer will be used by ruy kernel as a cache key
      _cached_weights = _weights->buffer();
    }
  }
#endif
#endif
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
