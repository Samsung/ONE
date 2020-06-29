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

#include "loader/KernelBuilder.h"

#include "kernels/Add.h"
#include "kernels/ArgMax.h"
#include "kernels/AveragePool2D.h"
#include "kernels/Concatenation.h"
#include "kernels/Conv2D.h"
#include "kernels/DepthwiseConv2D.h"
#include "kernels/Elu.h"
#include "kernels/FullyConnected.h"
#include "kernels/If.h"
#include "kernels/L2Normalize.h"
#include "kernels/L2Pool2D.h"
#include "kernels/LeakyRelu.h"
#include "kernels/LocalResponseNormalization.h"
#include "kernels/Logistic.h"
#include "kernels/MaxPool2D.h"
#include "kernels/Mean.h"
#include "kernels/Mul.h"
#include "kernels/Pad.h"
#include "kernels/Reshape.h"
#include "kernels/Softmax.h"
#include "kernels/Split.h"
#include "kernels/Unpack.h"
#include "kernels/Transpose.h"
#include "kernels/TransposeConv.h"
#include "loader/GraphLoader.h"
#include "loader/ModuleLoader.h"

#include <stdexcept>

namespace luci_interpreter
{

template <typename CircleNodeOut>
static std::vector<const loco::Node *> collectOutputNodes(const luci::CircleNode *node)
{
  std::vector<const CircleNodeOut *> output_nodes;
  for (const loco::Node *loco_node : loco::succs(node))
  {
    output_nodes.push_back(loco::must_cast<const CircleNodeOut *>(loco_node));
  }
  std::sort(output_nodes.begin(), output_nodes.end(),
            [](const CircleNodeOut *node1, const CircleNodeOut *node2) {
              return node1->index() < node2->index();
            });
  return {output_nodes.cbegin(), output_nodes.cend()};
}

const Tensor *KernelBuilder::getInputTensor(const loco::Node *node) const
{
  const Tensor *tensor = _graph_loader.getTensorForNode(node);
  assert(tensor != nullptr);
  return tensor;
}

const Tensor *KernelBuilder::getOptionalInputTensor(const loco::Node *node) const
{
  // TODO Revise this when optional inputs are implemented in the IR.
  return getInputTensor(node);
}

Tensor *KernelBuilder::getOutputTensor(const loco::Node *node) const
{
  Tensor *tensor = _graph_loader.getTensorForNode(node);
  assert(tensor != nullptr);
  return tensor;
}

std::vector<Tensor *>
KernelBuilder::getOutputTensors(const std::vector<const loco::Node *> &nodes) const
{
  std::vector<Tensor *> tensors;
  tensors.reserve(nodes.size());
  for (const loco::Node *node : nodes)
    tensors.push_back(getOutputTensor(node));
  return tensors;
}

RuntimeGraph *KernelBuilder::getRuntimeGraph(const loco::Graph *graph) const
{
  RuntimeGraph *runtime_graph = _module_loader.getRuntimeGraph(graph);
  assert(runtime_graph != nullptr);
  return runtime_graph;
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleAdd *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  AddParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Add>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleArgMax *node)
{
  assert(node->arity() == 2);
  if (dynamic_cast<const luci::CircleConst *>(node->dimension()) == nullptr)
    throw std::runtime_error("Dynamic dimension is not yet supported.");
  const Tensor *input1 = getInputTensor(node->input());
  const Tensor *input2 = getInputTensor(node->dimension());
  Tensor *output = getOutputTensor(node);

  ArgMaxParams params{};
  params.output_type = node->output_type();

  return std::make_unique<kernels::ArgMax>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleAveragePool2D *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->value());
  Tensor *output = getOutputTensor(node);

  Pool2DParams params{};
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

  return std::make_unique<kernels::Concatenation>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConst *)
{
  throw std::runtime_error("Const node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConv2D *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  Conv2DParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleDepthwiseConv2D *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  DepthwiseConv2DParams params{};
  params.padding = node->padding();
  params.depth_multiplier = node->depthMultiplier();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  // TODO Set dilations from the IR when it provides them.
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::DepthwiseConv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleElu *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Elu>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFullyConnected *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->weights());
  const Tensor *bias = getOptionalInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::FullyConnected>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleIf *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleIfOut>(node);
  assert(node->arity() == 1 + node->input_count());
  assert(output_nodes.size() == static_cast<size_t>(node->output_count()));

  const Tensor *cond = getInputTensor(node->cond());
  std::vector<const Tensor *> inputs(node->input_count());
  for (uint32_t i = 0; i < node->input_count(); ++i)
  {
    inputs[i] = getInputTensor(node->input(i));
  }
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  RuntimeGraph *then_graph = getRuntimeGraph(node->then_graph());
  RuntimeGraph *else_graph = getRuntimeGraph(node->else_graph());

  return std::make_unique<kernels::If>(cond, std::move(inputs), std::move(outputs), then_graph,
                                       else_graph);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleL2Normalize *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  L2NormParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Normalize>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleL2Pool2D *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->value());
  Tensor *output = getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Pool2D>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLeakyRelu *node)
{
  assert(node->arity() == 1);
  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  LeakyReluParams params{};
  params.alpha = node->alpha();

  return std::make_unique<kernels::LeakyRelu>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLocalResponseNormalization *node)
{
  assert(node->arity() == 1);
  const Tensor *input = getInputTensor(node->input());
  Tensor *output = getOutputTensor(node);

  LocalResponseNormalizationParams params{};
  params.radius = node->radius();
  params.bias = node->bias();
  params.alpha = node->alpha();
  params.beta = node->beta();

  return std::make_unique<kernels::LocalResponseNormalization>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogistic *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Logistic>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleInput *)
{
  throw std::runtime_error("Input node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMaxPool2D *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->value());
  Tensor *output = getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::MaxPool2D>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMean *node)
{
  assert(node->arity() == 2);

  if (dynamic_cast<const luci::CircleConst *>(node->reduction_indices()) == nullptr)
    throw std::runtime_error("Dynamic axes is not yet supported.");

  const Tensor *input = getInputTensor(node->input());
  const Tensor *axes = getInputTensor(node->reduction_indices());
  Tensor *output = getOutputTensor(node);

  ReducerParams params{};
  params.keep_dims = node->keep_dims();

  return std::make_unique<kernels::Mean>(input, axes, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMul *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  MulParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Mul>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleOutput *)
{
  throw std::runtime_error("Output node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CirclePad *node)
{
  assert(node->arity() == 2);

  if (dynamic_cast<const luci::CircleConst *>(node->paddings()) == nullptr)
    throw std::runtime_error("Dynamic padding is not yet supported.");

  const Tensor *input = getInputTensor(node->input());
  const Tensor *paddings = getInputTensor(node->paddings());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Pad>(input, paddings, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleReshape *node)
{
  assert(node->arity() == 2);

  if (dynamic_cast<const luci::CircleConst *>(node->shape()) == nullptr)
    throw std::runtime_error("Dynamic shape is not yet supported.");

  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *shape = getInputTensor(node->shape());
  Tensor *output = getOutputTensor(node);

  // NOTE 'newShape' attribute is ignored.
  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSoftmax *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  SoftmaxParams params{};
  params.beta = node->beta();

  return std::make_unique<kernels::Softmax>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSplit *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleSplitOut>(node);
  assert(node->arity() == 2);
  assert(output_nodes.size() == static_cast<size_t>(node->num_split()));

  if (dynamic_cast<const luci::CircleConst *>(node->split_dim()) == nullptr)
    throw std::runtime_error("Dynamic axis is not yet supported.");

  const Tensor *axis = getInputTensor(node->split_dim());
  const Tensor *input = getInputTensor(node->input());
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  // NOTE 'num_splits' attribute is ignored.
  return std::make_unique<kernels::Split>(axis, input, std::move(outputs));
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleTransposeConv *node)
{
  assert(node->arity() == 3);

  if (dynamic_cast<const luci::CircleConst *>(node->inputSizes()) == nullptr)
    throw std::runtime_error("Dynamic OutputShape is not yet supported.");

  const Tensor *input_sizes = getInputTensor(node->inputSizes());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *out_backprop = getInputTensor(node->outBackprop());

  Tensor *output = getOutputTensor(node);

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, output,
                                                  params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleUnpack *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleUnpackOut>(node);
  assert(node->arity() == 1);
  assert(output_nodes.size() == static_cast<size_t>(node->num()));

  const Tensor *input = getInputTensor(node->value());
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  UnpackParams params{};
  params.axis = node->axis();

  // NOTE 'num' attribute is ignored.
  return std::make_unique<kernels::Unpack>(input, std::move(outputs), params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleTranspose *node)
{
  assert(node->arity() == 2);

  if (dynamic_cast<const luci::CircleConst *>(node->perm()) == nullptr)
    throw std::runtime_error("Dynamic perm is not yet supported.");
  const Tensor *input = getInputTensor(node->a());
  const Tensor *perm = getInputTensor(node->perm());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Transpose>(input, perm, output);
}

} // namespace luci_interpreter
