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

namespace
{

using namespace onert;

std::unique_ptr<backend::train::Tensor>
createTransposedTensor(const backend::IPortableTensor *origin_tensor)
{
  const auto &origin_shape = origin_tensor->getShape();
  assert(origin_shape.rank() == 2);

  auto transposed_info = origin_tensor->get_info();
  auto transposed_shape = ir::Shape{origin_shape.dim(1), origin_shape.dim(0)};
  transposed_info.shape(transposed_shape);

  return std::make_unique<backend::train::Tensor>(transposed_info, origin_tensor->layout());
}

} // namespace

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
    _back_prop_input{nullptr}, _back_prop_output{nullptr}, _transposed_weights{nullptr},
    _transposed_input{nullptr}, _transposed_back_prop_output{nullptr},
    _act_back_prop_output{nullptr}
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::configureBackward(
  const IPortableTensor *input, const IPortableTensor *weights, IPortableTensor *output,
  IPortableTensor *back_prop_input, IPortableTensor *grad_weights, IPortableTensor *grad_bias,
  const IPortableTensor *back_prop_output, ir::Activation activation,
  ir::FullyConnectedWeightsFormat weights_format)
{
  _back_prop_input = back_prop_input;
  _grad_weights = grad_weights;
  _grad_bias = grad_bias;
  _back_prop_output = back_prop_output;

  if (weights_format != ir::FullyConnectedWeightsFormat::Default)
    throw std::runtime_error{
      "train FullyConnectedLayer: Weight formats other than default are not supported."};

  if (input->get_info().shape().rank() != 2 || weights->get_info().shape().rank() != 2 ||
      output->get_info().shape().rank() != 2 || back_prop_input->get_info().shape().rank() != 2 ||
      grad_weights->get_info().shape().rank() != 2 ||
      back_prop_output->get_info().shape().rank() != 2)
    throw std::runtime_error{
      "train FullyConnectedLayer: Input other ranks than 2 are not supported."};

  _transposed_weights = createTransposedTensor(weights);
  _transposed_weights->setBuffer(std::make_shared<basic::Allocator>(weights->total_size()));

  _transposed_input = createTransposedTensor(input);
  _transposed_input->setBuffer(std::make_shared<basic::Allocator>(input->total_size()));

  _transposed_back_prop_output = createTransposedTensor(back_prop_output);
  _transposed_back_prop_output->setBuffer(
    std::make_shared<basic::Allocator>(back_prop_output->total_size()));

  if (activation != ir::Activation::NONE)
  {
    _act_back_prop_output =
      std::make_unique<Tensor>(_back_prop_output->get_info(), _back_prop_output->layout());
    _act_back_prop_output->setBuffer(
      std::make_shared<basic::Allocator>(_back_prop_output->total_size()));
  }
}

void FullyConnectedLayer::forward(bool) { cpu::ops::FullyConnectedLayer::run(); }

void FullyConnectedLayer::backward()
{
  const auto data_type = _back_prop_output->data_type();
  assert(data_type == _input->data_type());
  switch (data_type)
  {
    case OperandType::FLOAT32:
    {
      assert(data_type == _grad_weights->data_type());
      assert(_grad_bias == nullptr || data_type == _grad_bias->data_type());
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
  try
  {
    backprop_act =
      backpropActivation(_activation, _output, _back_prop_output, _act_back_prop_output.get());
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error{"train FullyConnectedLayer: " + std::string(e.what())};
  }
  assert(backprop_act != nullptr);

  // Initialize TransposeParams
  nnfw::cker::TransposeParams transpose_param;
  transpose_param.perm_count = 2;
  transpose_param.perm[0] = 1;
  transpose_param.perm[1] = 0;

  // Initialize FullyConnectedParams
  nnfw::cker::FullyConnectedParams op_params;
  float output_activation_min = 0;
  float output_activation_max = 0;
  CalculateActivationRange(ir::Activation::NONE, &output_activation_min, &output_activation_max);
  op_params.activation = nnfw::cker::FusedActivationFunctionType::kNone;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  op_params.lhs_cacheable = false;
  op_params.rhs_cacheable = false;

  // Transpose and compute gradient for input
  // ∂L/∂X = fc(Incoming gradient, transposed W)
  auto transposed_weights = _transposed_weights.get();
  assert(transposed_weights->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(_weights), getBuffer<float>(_weights),
                        getShape(transposed_weights), getBuffer<float>(transposed_weights));

  nnfw::cker::FullyConnected(op_params, getShape(backprop_act), getBuffer<float>(backprop_act),
                             getShape(transposed_weights), getBuffer<float>(transposed_weights),
                             getShape(nullptr), nullptr, getShape(_back_prop_input),
                             getBuffer<float>(_back_prop_input));

  // Transpose and compute gradient for weights
  // ∂L/∂W = fc(transposed incomming gradient, transposed X)
  auto transposed_input = _transposed_input.get();
  assert(transposed_input->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(_input), getBuffer<float>(_input),
                        getShape(transposed_input), getBuffer<float>(transposed_input));

  auto transposed_back_prop_output = _transposed_back_prop_output.get();
  assert(transposed_back_prop_output->getShape().rank() == 2);
  nnfw::cker::Transpose(transpose_param, getShape(backprop_act), getBuffer<float>(backprop_act),
                        getShape(transposed_back_prop_output),
                        getBuffer<float>(transposed_back_prop_output));

  nnfw::cker::FullyConnected(
    op_params, getShape(transposed_back_prop_output), getBuffer<float>(transposed_back_prop_output),
    getShape(transposed_input), getBuffer<float>(transposed_input), getShape(nullptr), nullptr,
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
