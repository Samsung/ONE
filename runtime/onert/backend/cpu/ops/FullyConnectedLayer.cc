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

#include <cker/operation/FullyConnected.h>

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
      _activation(ir::Activation::NONE), _temp_arena(new nnfw::cker::FCTempArena())
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

  nnfw::cker::FullyConnectedHybrid(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_weights), reinterpret_cast<const int8_t *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()), temp_arena);
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    IPortableTensor *output)
{
  _input = input;
  _weights = weights;
  _bias = bias;
  _activation = activation;
  _output = output;
}

void FullyConnectedLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    if (_weights->data_type() == OperandType::QUANT_INT8_SYMM)
    {
      fullyConnectedHybrid();
    }
    else
    {
      fullyConnectedFloat32();
    }
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

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
