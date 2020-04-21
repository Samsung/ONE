/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "KernelBuilder.h"

#include "kernels/Add.h"
#include "kernels/AveragePool2D.h"
#include "kernels/Concatenation.h"
#include "kernels/Conv2D.h"
#include "kernels/DepthwiseConv2D.h"
#include "kernels/FullyConnected.h"
#include "kernels/MaxPool2D.h"
#include "kernels/Mul.h"
#include "kernels/Reshape.h"
#include "kernels/Softmax.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleAdd *node)
{
  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  AddParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Add>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleAveragePool2D *node)
{
  const Tensor *input = getInputTensor(node->value());
  Tensor *output = getOutputTensor(node);

  AveragePool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::AveragePool2D>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConcatenation *node)
{
  std::vector<const Tensor *> inputs(node->numValues());
  for (uint32_t i = 0; i < node->numValues(); ++i)
  {
    inputs[i] = getInputTensor(node->values(i));
  }
  Tensor *output = getOutputTensor(node);

  ConcatenationParams params{};
  params.axis = node->axis();

  return std::make_unique<kernels::Concatenation>(inputs, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConst *)
{
  throw std::logic_error("Const node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConv2D *node)
{
  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  Conv2DParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleDepthwiseConv2D *node)
{
  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  DepthwiseConv2DParams params{};
  params.padding = node->padding();
  params.depth_multiplier = node->depthMultiplier();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::DepthwiseConv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFullyConnected *node)
{
  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->weights());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::FullyConnected>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleInput *)
{
  throw std::logic_error("Input node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMaxPool2D *node)
{
  const Tensor *input = getInputTensor(node->value());
  Tensor *output = getOutputTensor(node);

  MaxPool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::MaxPool2D>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMul *node)
{
  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  MulParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Mul>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleOutput *)
{
  throw std::logic_error("Output node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleReshape *node)
{
  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *shape = getInputTensor(node->shape());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSoftmax *node)
{
  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  SoftmaxParams params{};
  params.beta = node->beta();

  return std::make_unique<kernels::Softmax>(input, output, params);
}

} // namespace luci_interpreter
