/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <ruy/operation/FullyConnected.h>
#include <ruy/TensorUtils.h>

namespace onert
{
namespace backend
{
namespace ruy
{
namespace ops
{

FullyConnectedLayer::FullyConnectedLayer()
    : _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr),
      _activation(ir::Activation::NONE), _external_context(nullptr)
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::fullyConnectedFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);
  nnfw::ruy::FullyConnectedParams op_params;

  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  op_params.activation = convertActivationType(_activation);
  op_params.lhs_cacheable = _weights->is_constant();
  op_params.rhs_cacheable = _input->is_constant();

  nnfw::ruy::FullyConnected(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_weights), reinterpret_cast<const float *>(_weights->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias ? _bias->buffer() : nullptr),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()),
      _external_context->ruy_context());
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    ir::FullyConnectedWeightsFormat weights_format,
                                    IPortableTensor *output,
                                    const std::shared_ptr<ExternalContext> &external_context)
{
  UNUSED_RELEASE(weights_format);
  _input = input;
  _weights = weights;
  _bias = bias;
  _activation = activation;
  _output = output;
  _external_context = external_context;
}

void FullyConnectedLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    fullyConnectedFloat32();
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
    if (nnfw::ruy::IsZeroVector(reinterpret_cast<float *>(_bias->buffer()), bias_size))
    {
      _bias = nullptr;
    }
  }
}

} // namespace ops
} // namespace ruy
} // namespace backend
} // namespace onert
