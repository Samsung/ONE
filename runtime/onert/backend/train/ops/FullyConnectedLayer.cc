/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OperationUtils.h"

#include <cker/operation/FullyConnected.h>
#include <cker/operation/Transpose.h>
#include <cker/train/operation/FullyConnected.h>
#include <cker/train/operation/ReLU.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

FullyConnectedLayer::FullyConnectedLayer()
  : cpu::ops::FullyConnectedLayer{}, _grad_weights{nullptr}, _grad_bias{nullptr},
    _deriv_input{nullptr}, _deriv_output{nullptr}, _transposed_weights{nullptr},
    _transposed_input{nullptr}, _transposed_deriv_output{nullptr}, _act_deriv_output{nullptr}
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, IPortableTensor *output,
                                    IPortableTensor *deriv_input, IPortableTensor *grad_weights,
                                    IPortableTensor *grad_bias, const IPortableTensor *deriv_output,
                                    ir::Activation activation,
                                    ir::FullyConnectedWeightsFormat weights_format,
                                    const std::shared_ptr<train::ExternalContext> &external_context)
{
  cpu::ops::FullyConnectedLayer::configure(input, weights, bias, activation, weights_format, output,
                                           external_context);

  _deriv_input = deriv_input;
  _grad_weights = grad_weights;
  _grad_bias = grad_bias;
  _deriv_output = deriv_output;

  if (weights_format != ir::FullyConnectedWeightsFormat::Default)
    throw std::runtime_error{
      "train FullyConnectedLayer: Weight formats other than default are not supported."};

  if (input->get_info().shape().rank() != 2 || weights->get_info().shape().rank() != 2 ||
      output->get_info().shape().rank() != 2 || deriv_input->get_info().shape().rank() != 2 ||
      grad_weights->get_info().shape().rank() != 2 || deriv_output->get_info().shape().rank() != 2)
    throw std::runtime_error{
      "train FullyConnectedLayer: Input other ranks than 2 are not supported."};

  _transposed_weights = std::make_unique<Tensor>(weights->get_info(), weights->layout());
  _transposed_weights->setBuffer(std::make_shared<basic::Allocator>(weights->total_size()));

  _transposed_input = std::make_unique<Tensor>(input->get_info(), input->layout());
  _transposed_input->setBuffer(std::make_shared<basic::Allocator>(input->total_size()));

  _transposed_deriv_output =
    std::make_unique<Tensor>(deriv_output->get_info(), deriv_output->layout());
  _transposed_deriv_output->setBuffer(
    std::make_shared<basic::Allocator>(deriv_output->total_size()));

  if (activation != ir::Activation::NONE)
  {
    _act_deriv_output =
      std::make_unique<Tensor>(_deriv_output->get_info(), _deriv_output->layout());
    _act_deriv_output->setBuffer(std::make_shared<basic::Allocator>(_deriv_output->total_size()));
  }
}

void FullyConnectedLayer::forward(bool) { cpu::ops::FullyConnectedLayer::run(); }

void FullyConnectedLayer::backward(uint32_t)
{
  const auto data_type = _deriv_output->data_type();
  assert(data_type == _input->data_type());
  switch (data_type)
  {
    case OperandType::FLOAT32:
    {
      assert(data_type == _grad_weights->data_type());
      assert(data_type == _grad_bias->data_type());
      backwardFloat32();
      break;
    }
    default:
      throw std::runtime_error{"train FullyConnectedLayer: unsupported data type"};
  }
}

void FullyConnectedLayer::backwardFloat32()
{
  // Calculate gradient for activation
  const IPortableTensor *backprop_act;
  switch (_activation)
  {
    case ir::Activation::NONE:
      backprop_act = _deriv_output;
      break;
    case ir::Activation::RELU:
      nnfw::cker::train::ReLUGrad(getShape(_output), getBuffer<float>(_output),
                                  getShape(_deriv_output), getBuffer<float>(_deriv_output),
                                  getShape(_act_deriv_output.get()),
                                  getBuffer<float>(_act_deriv_output.get()));
      backprop_act = _act_deriv_output.get();
      break;
    default:
      throw std::runtime_error("train FullyConnectedLayer: Unsupported activation type yet");
  }

  // Initialize TransposeParams
  nnfw::cker::TransposeParams transpose_param;
  transpose_param.perm_count = 2;
  transpose_param.perm[0] = 1;
  transpose_param.perm[1] = 0;

  // Initialize FullyConnectedParams
  nnfw::cker::FullyConnectedParams op_params;
  op_params.activation = nnfw::cker::FusedActivationFunctionType::kNone;

  // Transpose and compute gradient for input
  // ∂L/∂X = fc(Incoming gradient, transposed W)
  auto transposed_weights = _transposed_weights.get();
  assert(transposed_weights->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(_weights), getBuffer<float>(_weights),
                        getShape(transposed_weights), getBuffer<float>(transposed_weights));

  nnfw::cker::FullyConnected(op_params, getShape(backprop_act), getBuffer<float>(backprop_act),
                             getShape(transposed_weights), getBuffer<float>(transposed_weights),
                             getShape(nullptr), nullptr, getShape(_deriv_input),
                             getBuffer<float>(_deriv_input));

  // Transpose and compute gradient for weights
  // ∂L/∂W = fc(transposed incomming gradient, transposed X)
  auto transposed_input = _transposed_input.get();
  assert(transposed_input->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(_input), getBuffer<float>(_input),
                        getShape(transposed_input), getBuffer<float>(transposed_input));

  auto transposed_deriv_output = _transposed_deriv_output.get();
  assert(transposed_deriv_output->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(backprop_act), getBuffer<float>(backprop_act),
                        getShape(transposed_deriv_output),
                        getBuffer<float>(transposed_deriv_output));

  nnfw::cker::FullyConnected(op_params, getShape(transposed_deriv_output),
                             getBuffer<float>(transposed_deriv_output), getShape(transposed_input),
                             getBuffer<float>(transposed_input), getShape(nullptr), nullptr,
                             getShape(_grad_weights), getBuffer<float>(_grad_weights));

  // Compute gradient for bias
  if (_bias)
  {
    assert(_grad_bias);
    nnfw::cker::train::FullyConnectedBiasGrad(getShape(backprop_act),
                                              getBuffer<float>(backprop_act), getShape(_grad_bias),
                                              getBuffer<float>(_grad_bias));
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
