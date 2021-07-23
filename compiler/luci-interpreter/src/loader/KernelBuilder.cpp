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
#include "kernels/BatchToSpaceND.h"
#include "kernels/Concatenation.h"
#include "kernels/Conv2D.h"
#include "kernels/DepthToSpace.h"
#include "kernels/DepthwiseConv2D.h"
#include "kernels/Div.h"
#include "kernels/Elu.h"
#include "kernels/Exp.h"
#include "kernels/Floor.h"
#include "kernels/FloorDiv.h"
#include "kernels/Equal.h"
#include "kernels/FullyConnected.h"
#include "kernels/Greater.h"
#include "kernels/GreaterEqual.h"
#include "kernels/If.h"
#include "kernels/InstanceNorm.h"
#include "kernels/L2Normalize.h"
#include "kernels/L2Pool2D.h"
#include "kernels/LeakyRelu.h"
#include "kernels/Less.h"
#include "kernels/LessEqual.h"
#include "kernels/LocalResponseNormalization.h"
#include "kernels/LogicalAnd.h"
#include "kernels/LogicalNot.h"
#include "kernels/LogicalOr.h"
#include "kernels/Logistic.h"
#include "kernels/LogSoftmax.h"
#include "kernels/Maximum.h"
#include "kernels/MaxPool2D.h"
#include "kernels/Mean.h"
#include "kernels/Minimum.h"
#include "kernels/MirrorPad.h"
#include "kernels/Mul.h"
#include "kernels/Neg.h"
#include "kernels/NotEqual.h"
#include "kernels/Pack.h"
#include "kernels/Pad.h"
#include "kernels/PadV2.h"
#include "kernels/Pow.h"
#include "kernels/Prelu.h"
#include "kernels/Relu.h"
#include "kernels/Relu6.h"
#include "kernels/Reshape.h"
#include "kernels/ResizeBilinear.h"
#include "kernels/ResizeNearestNeighbor.h"
#include "kernels/Reverse.h"
#include "kernels/Rsqrt.h"
#include "kernels/Slice.h"
#include "kernels/Softmax.h"
#include "kernels/SpaceToBatchND.h"
#include "kernels/SpaceToDepth.h"
#include "kernels/Split.h"
#include "kernels/StridedSlice.h"
#include "kernels/Sqrt.h"
#include "kernels/Square.h"
#include "kernels/SquaredDifference.h"
#include "kernels/Squeeze.h"
#include "kernels/Sub.h"
#include "kernels/Tanh.h"
#include "kernels/Unpack.h"
#include "kernels/Transpose.h"
#include "kernels/TransposeConv.h"

#include <stdexcept>

namespace
{

template <typename CircleNodeOut>
std::vector<const loco::Node *> collectOutputNodes(const luci::CircleNode *node)
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

} // namespace

namespace luci_interpreter
{

namespace
{

std::unique_ptr<Kernel> build_kernel_CircleAdd(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleAdd *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  AddParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Add>(input1, input2, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleArgMax(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleArgMax *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);
  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *axis = helper.getInputTensor(node->dimension());
  Tensor *output = helper.getOutputTensor(node);

  ArgMaxParams params{};
  params.output_type = node->output_type();

  return std::make_unique<kernels::ArgMax>(input, axis, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleAveragePool2D(const luci::CircleNode *circle_node,
                                                         KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleAveragePool2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->value());
  Tensor *output = helper.getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::AveragePool2D>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleBatchToSpaceND(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleBatchToSpaceND *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *block_shape = helper.getInputTensor(node->block_shape());
  const Tensor *crops = helper.getInputTensor(node->crops());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::BatchToSpaceND>(input, block_shape, crops, output);
}

std::unique_ptr<Kernel> build_kernel_CircleConcatenation(const luci::CircleNode *circle_node,
                                                         KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleConcatenation *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  std::vector<const Tensor *> inputs(node->numValues());
  for (uint32_t i = 0; i < node->numValues(); ++i)
  {
    inputs[i] = helper.getInputTensor(node->values(i));
  }
  Tensor *output = helper.getOutputTensor(node);

  ConcatenationParams params{};
  params.axis = node->axis();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Concatenation>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleConv2D(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleConv2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *bias = helper.getInputTensor(node->bias());
  Tensor *output = helper.getOutputTensor(node);

  Conv2DParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleDepthToSpace(const luci::CircleNode *circle_node,
                                                        KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleDepthToSpace *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->input());
  Tensor *output = helper.getOutputTensor(node);

  DepthToSpaceParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::DepthToSpace>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleDepthwiseConv2D(const luci::CircleNode *circle_node,
                                                           KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleDepthwiseConv2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *bias = helper.getInputTensor(node->bias());
  Tensor *output = helper.getOutputTensor(node);

  DepthwiseConv2DParams params{};
  params.padding = node->padding();
  params.depth_multiplier = node->depthMultiplier();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::DepthwiseConv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleDiv(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleDiv *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);
  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  DivParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Div>(input1, input2, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleElu(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleElu *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->features());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Elu>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleEqual(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleEqual *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Equal>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CircleExp(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleExp *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Exp>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleFloor(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleFloor *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Floor>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleFloorDiv(const luci::CircleNode *circle_node,
                                                    KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleFloorDiv *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::FloorDiv>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CircleFullyConnected(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleFullyConnected *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *weights = helper.getInputTensor(node->weights());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());
  Tensor *output = helper.getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleGreater(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleGreater *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Greater>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CircleGreaterEqual(const luci::CircleNode *circle_node,
                                                        KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleGreaterEqual *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::GreaterEqual>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CircleIf(const luci::CircleNode *circle_node,
                                              KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleIf *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  auto output_nodes = collectOutputNodes<luci::CircleIfOut>(node);
  assert(node->arity() == 1 + node->input_count());
  assert(output_nodes.size() == static_cast<size_t>(node->output_count()));

  const Tensor *cond = helper.getInputTensor(node->cond());
  std::vector<const Tensor *> inputs(node->input_count());
  for (uint32_t i = 0; i < node->input_count(); ++i)
  {
    inputs[i] = helper.getInputTensor(node->input(i));
  }
  std::vector<Tensor *> outputs = helper.getOutputTensors(output_nodes);

  RuntimeGraph *then_graph = helper.getRuntimeGraph(node->then_graph());
  RuntimeGraph *else_graph = helper.getRuntimeGraph(node->else_graph());

  return std::make_unique<kernels::If>(cond, std::move(inputs), std::move(outputs), then_graph,
                                       else_graph);
}

std::unique_ptr<Kernel> build_kernel_CircleInstanceNorm(const luci::CircleNode *circle_node,
                                                        KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleInstanceNorm *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *gamma = helper.getInputTensor(node->gamma());
  const Tensor *beta = helper.getInputTensor(node->beta());

  Tensor *output = helper.getOutputTensor(node);

  InstanceNormParams params{};
  params.epsilon = node->epsilon();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::InstanceNorm>(input, gamma, beta, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleL2Normalize(const luci::CircleNode *circle_node,
                                                       KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleL2Normalize *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  L2NormParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Normalize>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleL2Pool2D(const luci::CircleNode *circle_node,
                                                    KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleL2Pool2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->value());
  Tensor *output = helper.getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Pool2D>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleLeakyRelu(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLeakyRelu *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);
  const Tensor *input = helper.getInputTensor(node->features());
  Tensor *output = helper.getOutputTensor(node);

  LeakyReluParams params{};
  params.alpha = node->alpha();

  return std::make_unique<kernels::LeakyRelu>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleLess(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLess *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Less>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CircleLessEqual(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLessEqual *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::LessEqual>(x, y, output);
}

std::unique_ptr<Kernel>
build_kernel_CircleLocalResponseNormalization(const luci::CircleNode *circle_node,
                                              KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLocalResponseNormalization *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);
  const Tensor *input = helper.getInputTensor(node->input());
  Tensor *output = helper.getOutputTensor(node);

  LocalResponseNormalizationParams params{};
  params.radius = node->radius();
  params.bias = node->bias();
  params.alpha = node->alpha();
  params.beta = node->beta();

  return std::make_unique<kernels::LocalResponseNormalization>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleLogicalAnd(const luci::CircleNode *circle_node,
                                                      KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLogicalAnd *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::LogicalAnd>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CircleLogicalNot(const luci::CircleNode *circle_node,
                                                      KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLogicalNot *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::LogicalNot>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleLogicalOr(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLogicalOr *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::LogicalOr>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CircleLogistic(const luci::CircleNode *circle_node,
                                                    KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLogistic *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Logistic>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleLogSoftmax(const luci::CircleNode *circle_node,
                                                      KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleLogSoftmax *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->logits());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::LogSoftmax>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleMaximum(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMaximum *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Maximum>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CircleMaxPool2D(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMaxPool2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->value());
  Tensor *output = helper.getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::MaxPool2D>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleMean(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMean *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *axes = helper.getInputTensor(node->reduction_indices());
  Tensor *output = helper.getOutputTensor(node);

  ReducerParams params{};
  params.keep_dims = node->keep_dims();

  return std::make_unique<kernels::Mean>(input, axes, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleMinimum(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMinimum *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Minimum>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CircleMirrorPad(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMirrorPad *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *paddings = helper.getInputTensor(node->paddings());
  Tensor *output = helper.getOutputTensor(node);

  MirrorPadParams params{};
  params.mode = node->mode();

  return std::make_unique<kernels::MirrorPad>(input, paddings, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleMul(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleMul *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  MulParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Mul>(input1, input2, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleNeg(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleNeg *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Neg>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleNotEqual(const luci::CircleNode *circle_node,
                                                    KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleNotEqual *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *x = helper.getInputTensor(node->x());
  const Tensor *y = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::NotEqual>(x, y, output);
}

std::unique_ptr<Kernel> build_kernel_CirclePack(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CirclePack *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == node->values_count());

  std::vector<const Tensor *> inputs(node->values_count());
  for (uint32_t i = 0; i < node->values_count(); ++i)
  {
    inputs[i] = helper.getInputTensor(node->values(i));
  }
  Tensor *output = helper.getOutputTensor(node);

  PackParams params{};
  params.axis = node->axis();
  params.values_count = node->values_count();

  return std::make_unique<kernels::Pack>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> build_kernel_CirclePad(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CirclePad *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *paddings = helper.getInputTensor(node->paddings());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Pad>(input, paddings, output);
}

std::unique_ptr<Kernel> build_kernel_CirclePadV2(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CirclePadV2 *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *paddings = helper.getInputTensor(node->paddings());
  const Tensor *constant_values = helper.getInputTensor(node->constant_values());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::PadV2>(input, paddings, constant_values, output);
}

std::unique_ptr<Kernel> build_kernel_CirclePow(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CirclePow *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());

  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Pow>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CirclePRelu(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CirclePRelu *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *alpha = helper.getInputTensor(node->alpha());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Prelu>(input, alpha, output);
}

std::unique_ptr<Kernel> build_kernel_CircleRelu(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleRelu *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->features());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Relu>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleRelu6(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleRelu6 *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->features());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Relu6>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleReshape(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleReshape *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->tensor());
  const Tensor *shape = helper.getInputTensor(node->shape());
  Tensor *output = helper.getOutputTensor(node);

  // NOTE 'newShape' attribute is ignored.
  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> build_kernel_CircleResizeBilinear(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleResizeBilinear *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *size = helper.getInputTensor(node->size());
  Tensor *output = helper.getOutputTensor(node);

  ResizeBilinearParams params{};
  params.align_corners = node->align_corners();
  params.half_pixel_centers = node->half_pixel_centers();

  return std::make_unique<kernels::ResizeBilinear>(input, size, output, params);
}

std::unique_ptr<Kernel>
build_kernel_CircleResizeNearestNeighbor(const luci::CircleNode *circle_node,
                                         KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleResizeNearestNeighbor *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *size = helper.getInputTensor(node->size());
  Tensor *output = helper.getOutputTensor(node);

  ResizeNearestNeighborParams params{};
  params.align_corners = node->align_corners();
  // TODO update half_pixel_centers after CircleResizeNearestNeighbor updated
  // Current CircleResizeNearestNeighbor don't have half_pixel_centers.
  // default value on current is false.
  // it need to be updated when CircleResizeNearestNeighbor updated.
  params.half_pixel_centers = false;

  return std::make_unique<kernels::ResizeNearestNeighbor>(input, size, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleReverseV2(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleReverseV2 *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->tensor());
  const Tensor *axes = helper.getInputTensor(node->axis());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Reverse>(input, axes, output);
}

std::unique_ptr<Kernel> build_kernel_CircleRsqrt(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleRsqrt *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Rsqrt>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSlice(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSlice *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *begin = helper.getInputTensor(node->begin());
  const Tensor *size = helper.getInputTensor(node->size());

  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Slice>(input, begin, size, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSoftmax(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSoftmax *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->logits());
  Tensor *output = helper.getOutputTensor(node);

  SoftmaxParams params{};
  params.beta = node->beta();

  return std::make_unique<kernels::Softmax>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleSpaceToBatchND(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSpaceToBatchND *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *block_shape = helper.getInputTensor(node->block_shape());
  const Tensor *paddings = helper.getInputTensor(node->paddings());

  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::SpaceToBatchND>(input, block_shape, paddings, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSpaceToDepth(const luci::CircleNode *circle_node,
                                                        KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSpaceToDepth *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);
  const Tensor *input = helper.getInputTensor(node->input());

  Tensor *output = helper.getOutputTensor(node);

  SpaceToDepthParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::SpaceToDepth>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleSplit(const luci::CircleNode *circle_node,
                                                 KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSplit *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  auto output_nodes = collectOutputNodes<luci::CircleSplitOut>(node);
  assert(node->arity() == 2);
  assert(output_nodes.size() == static_cast<size_t>(node->num_split()));

  const Tensor *axis = helper.getInputTensor(node->split_dim());
  const Tensor *input = helper.getInputTensor(node->input());
  std::vector<Tensor *> outputs = helper.getOutputTensors(output_nodes);

  // NOTE 'num_splits' attribute is ignored.
  return std::make_unique<kernels::Split>(axis, input, std::move(outputs));
}

std::unique_ptr<Kernel> build_kernel_CircleSqrt(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSqrt *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Sqrt>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSquare(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSquare *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Square>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSquaredDifference(const luci::CircleNode *circle_node,
                                                             KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSquaredDifference *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::SquaredDifference>(input1, input2, output);
}

std::unique_ptr<Kernel> build_kernel_CircleSqueeze(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSqueeze *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->input());
  Tensor *output = helper.getOutputTensor(node);

  SqueezeParams params{};
  params.squeeze_dims = node->squeeze_dims();

  return std::make_unique<kernels::Squeeze>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleStridedSlice(const luci::CircleNode *circle_node,
                                                        KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleStridedSlice *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 4);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *begin = helper.getInputTensor(node->begin());
  const Tensor *end = helper.getInputTensor(node->end());
  const Tensor *strides = helper.getInputTensor(node->strides());

  Tensor *output = helper.getOutputTensor(node);

  StridedSliceParams params{};
  params.begin_mask = node->begin_mask();
  params.ellipsis_mask = node->ellipsis_mask();
  params.end_mask = node->end_mask();
  params.new_axis_mask = node->new_axis_mask();
  params.shrink_axis_mask = node->shrink_axis_mask();

  return std::make_unique<kernels::StridedSlice>(input, begin, end, strides, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleSub(const luci::CircleNode *circle_node,
                                               KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleSub *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input1 = helper.getInputTensor(node->x());
  const Tensor *input2 = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  SubParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Sub>(input1, input2, output, params);
}

std::unique_ptr<Kernel> build_kernel_CircleTanh(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleTanh *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->x());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Tanh>(input, output);
}

std::unique_ptr<Kernel> build_kernel_CircleTranspose(const luci::CircleNode *circle_node,
                                                     KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleTranspose *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->a());
  const Tensor *perm = helper.getInputTensor(node->perm());
  Tensor *output = helper.getOutputTensor(node);

  return std::make_unique<kernels::Transpose>(input, perm, output);
}

std::unique_ptr<Kernel> build_kernel_CircleTransposeConv(const luci::CircleNode *circle_node,
                                                         KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleTransposeConv *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 4);

  const Tensor *input_sizes = helper.getInputTensor(node->inputSizes());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *out_backprop = helper.getInputTensor(node->outBackprop());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());

  Tensor *output = helper.getOutputTensor(node);

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
                                                  params);
}

std::unique_ptr<Kernel> build_kernel_CircleUnpack(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleUnpack *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  auto output_nodes = collectOutputNodes<luci::CircleUnpackOut>(node);
  assert(node->arity() == 1);
  assert(output_nodes.size() == static_cast<size_t>(node->num()));

  const Tensor *input = helper.getInputTensor(node->value());
  std::vector<Tensor *> outputs = helper.getOutputTensors(output_nodes);

  UnpackParams params{};
  params.axis = node->axis();

  // NOTE 'num' attribute is ignored.
  return std::make_unique<kernels::Unpack>(input, std::move(outputs), params);
}

} // namespace

#define CIRCLE_NODE(OPCODE, CLASS) CLASS,
#define CIRCLE_VNODE(OPCODE, CLASS) CLASS,

// This enum is auxiliary.
// It is duplicate of luci::CircleOpcode but initialized with CLASS instead of OPCODE,
// because list of target operators is in format of CLASS names
enum class BuilderId
{
#include <luci/IR/CircleNodes.lst>
};

#undef CIRCLE_VNODE
#undef CIRCLE_NODE

class KernelBuilderRegistry
{
  using KernelBuilderFunc = std::unique_ptr<Kernel>(const luci::CircleNode *,
                                                    KernelBuilderHelper &);

public:
  KernelBuilderRegistry(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : _helper(graph_to_runtime_graph, node_to_tensor)
  {
#define REGISTER_KERNEL(name) \
 register_kernel_builder(BuilderId::Circle##name, build_kernel_Circle##name);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL
  }

  std::unique_ptr<Kernel> build_kernel(const luci::CircleNode *node)
  {
    auto opcode = static_cast<int>(node->opcode());
    if (opcode >= static_cast<int64_t>(_operator_builders.size()))
      return nullptr;
    if (_operator_builders[opcode] == nullptr)
      return nullptr;
    return _operator_builders[opcode](node, _helper);
  }

private:
  KernelBuilderHelper _helper;

  std::vector<KernelBuilderFunc *> _operator_builders;

  void register_kernel_builder(BuilderId id, KernelBuilderFunc *func)
  {
    auto opcode = static_cast<int>(id);
    if (opcode >= static_cast<int64_t>(_operator_builders.size()))
      _operator_builders.resize(opcode + 1);
    _operator_builders[opcode] = func;
  }
};

KernelBuilder::KernelBuilder(const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph, const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor): KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
{
  _builder_registry = std::make_unique<KernelBuilderRegistry>(graph_to_runtime_graph, node_to_tensor);
}

KernelBuilder::~KernelBuilder()
{
  // Need to define in this CPP to hide KernelBuilderRegistry internals.
  // This destructor deletes _builder_registry
}

std::unique_ptr<Kernel> KernelBuilder::build(const luci::CircleNode *node)
{
  auto ret = _builder_registry->build_kernel(node);
  if (ret != nullptr)
    return ret;

  std::string msg = "Unsupported operator: ";
  msg += std::to_string(static_cast<uint32_t>(node->opcode())) + " " + std::string(node->name());
  throw std::invalid_argument(msg.c_str());
}

} // namespace luci_interpreter
