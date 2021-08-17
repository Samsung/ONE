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
#include <luci_interpreter/CircleNodeMemoryPlan.h>

#include "kernels/Add.h"
#include "kernels/ArgMax.h"
#include "kernels/AveragePool2D.h"
#include "kernels/BatchToSpaceND.h"
#include "kernels/Cast.h"
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
#include "kernels/PRelu.h"
#include "kernels/Relu.h"
#include "kernels/Relu6.h"
#include "kernels/Reshape.h"
#include "kernels/ResizeBilinear.h"
#include "kernels/ResizeNearestNeighbor.h"
#include "kernels/ReverseV2.h"
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
#include "kernels/While.h"

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

// TODO move to anonymous namespace
enum class KB
{
  ABC,
  DEF,
  GHIJ,
  KLMN,
  OPQR,
  STUV,
  WXYZ,
};

#define DECLARE_VISIT(CLASS) std::unique_ptr<Kernel> visit(const luci::CLASS *) override

template <KB kb> class KernelBuilderLet;

template <>
class KernelBuilderLet<KB::ABC> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                  public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleAdd);
  DECLARE_VISIT(CircleArgMax);
  DECLARE_VISIT(CircleAveragePool2D);
  DECLARE_VISIT(CircleBatchToSpaceND);
  DECLARE_VISIT(CircleCast);
  DECLARE_VISIT(CircleConcatenation);
  DECLARE_VISIT(CircleConst);
  DECLARE_VISIT(CircleConv2D);
};

template <>
class KernelBuilderLet<KB::DEF> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                  public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleDepthToSpace);
  DECLARE_VISIT(CircleDepthwiseConv2D);
  DECLARE_VISIT(CircleDiv);
  DECLARE_VISIT(CircleElu);
  DECLARE_VISIT(CircleEqual);
  DECLARE_VISIT(CircleExp);
  DECLARE_VISIT(CircleFloor);
  DECLARE_VISIT(CircleFloorDiv);
  DECLARE_VISIT(CircleFullyConnected);
};

template <>
class KernelBuilderLet<KB::GHIJ> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                   public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleGreater);
  DECLARE_VISIT(CircleGreaterEqual);
  DECLARE_VISIT(CircleIf);
  DECLARE_VISIT(CircleInput);
  DECLARE_VISIT(CircleInstanceNorm);
};

template <>
class KernelBuilderLet<KB::KLMN> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                   public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleL2Normalize);
  DECLARE_VISIT(CircleL2Pool2D);
  DECLARE_VISIT(CircleLeakyRelu);
  DECLARE_VISIT(CircleLess);
  DECLARE_VISIT(CircleLessEqual);
  DECLARE_VISIT(CircleLocalResponseNormalization);
  DECLARE_VISIT(CircleLogSoftmax);
  DECLARE_VISIT(CircleLogicalAnd);
  DECLARE_VISIT(CircleLogicalNot);
  DECLARE_VISIT(CircleLogicalOr);
  DECLARE_VISIT(CircleLogistic);
  DECLARE_VISIT(CircleMaxPool2D);
  DECLARE_VISIT(CircleMaximum);
  DECLARE_VISIT(CircleMean);
  DECLARE_VISIT(CircleMinimum);
  DECLARE_VISIT(CircleMirrorPad);
  DECLARE_VISIT(CircleMul);
  DECLARE_VISIT(CircleNeg);
  DECLARE_VISIT(CircleNotEqual);
};

template <>
class KernelBuilderLet<KB::OPQR> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                   public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleOutput);
  DECLARE_VISIT(CirclePRelu);
  DECLARE_VISIT(CirclePack);
  DECLARE_VISIT(CirclePad);
  DECLARE_VISIT(CirclePadV2);
  DECLARE_VISIT(CirclePow);
  DECLARE_VISIT(CircleRelu);
  DECLARE_VISIT(CircleRelu6);
  DECLARE_VISIT(CircleReshape);
  DECLARE_VISIT(CircleResizeBilinear);
  DECLARE_VISIT(CircleResizeNearestNeighbor);
  DECLARE_VISIT(CircleReverseV2);
  DECLARE_VISIT(CircleRsqrt);
};

template <>
class KernelBuilderLet<KB::STUV> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                   public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleSlice);
  DECLARE_VISIT(CircleSoftmax);
  DECLARE_VISIT(CircleSpaceToBatchND);
  DECLARE_VISIT(CircleSpaceToDepth);
  DECLARE_VISIT(CircleSplit);
  DECLARE_VISIT(CircleSqrt);
  DECLARE_VISIT(CircleSquare);
  DECLARE_VISIT(CircleSquaredDifference);
  DECLARE_VISIT(CircleSqueeze);
  DECLARE_VISIT(CircleStridedSlice);
  DECLARE_VISIT(CircleSub);
  DECLARE_VISIT(CircleTanh);
  DECLARE_VISIT(CircleTranspose);
  DECLARE_VISIT(CircleTransposeConv);
  DECLARE_VISIT(CircleUnpack);
};

template <>
class KernelBuilderLet<KB::WXYZ> : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                                   public KernelBuilderHelper
{
public:
  KernelBuilderLet(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

public:
  std::unique_ptr<Kernel> visit(const luci::CircleNode *) { return nullptr; }

public:
  DECLARE_VISIT(CircleWhile);
};

#undef DECLARE_VISIT

std::unique_ptr<Kernel> KernelBuilder::build(const luci::CircleNode *node)
{
#define VISIT_KB(GRP)                                                          \
  do                                                                           \
  {                                                                            \
    KernelBuilderLet<KB::GRP> kbl(graph_to_runtime_graph(), node_to_tensor()); \
    auto ret = node->accept(&kbl);                                             \
    if (ret != nullptr)                                                        \
      return ret;                                                              \
  } while (false)

  VISIT_KB(ABC);
  VISIT_KB(DEF);
  VISIT_KB(GHIJ);
  VISIT_KB(KLMN);
  VISIT_KB(OPQR);
  VISIT_KB(STUV);
  VISIT_KB(WXYZ);

#undef VISIT_KB
  std::string msg = "Unsupported operator: ";
  msg += std::to_string(static_cast<uint32_t>(node->opcode())) + " " + std::string(node->name());
  throw std::invalid_argument(msg.c_str());
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleAdd *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  AddParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Add>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleArgMax *node)
{
  assert(node->arity() == 2);
  const Tensor *input = getInputTensor(node->input());
  const Tensor *axis = getInputTensor(node->dimension());
  Tensor *output = getOutputTensor(node);

  ArgMaxParams params{};
  params.output_type = node->output_type();

  return std::make_unique<kernels::ArgMax>(input, axis, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleAveragePool2D *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleBatchToSpaceND *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *block_shape = getInputTensor(node->block_shape());
  const Tensor *crops = getInputTensor(node->crops());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::BatchToSpaceND>(input, block_shape, crops, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleCast *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Cast>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleConcatenation *node)
{
  std::vector<const Tensor *> inputs(node->numValues());
  for (uint32_t i = 0; i < node->numValues(); ++i)
  {
    inputs[i] = getInputTensor(node->values(i));
  }
  Tensor *output = getOutputTensor(node);

  ConcatenationParams params{};
  params.axis = node->axis();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Concatenation>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleConst *)
{
  throw std::runtime_error("Const node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::ABC>::visit(const luci::CircleConv2D *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *bias = getInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  auto im2col =
    std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  im2col->make_unobservable();
  im2col->set_data_buffer(nullptr);
  auto memory_plan = luci::get_memory_plan(node);
  im2col->set_offset(memory_plan.offset()[1]);
  Tensor *tmp = getRuntimeGraph(node->graph())->addTensor(std::move(im2col));

  Conv2DParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, tmp, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleDepthToSpace *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->input());
  Tensor *output = getOutputTensor(node);

  DepthToSpaceParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::DepthToSpace>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleDepthwiseConv2D *node)
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
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::DepthwiseConv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleDiv *node)
{
  assert(node->arity() == 2);
  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  DivParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Div>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleElu *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Elu>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Equal>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleExp *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Exp>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleFloor *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Floor>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleFloorDiv *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::FloorDiv>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::DEF>::visit(const luci::CircleFullyConnected *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *weights = getInputTensor(node->weights());
  const Tensor *bias = getOptionalInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::GHIJ>::visit(const luci::CircleGreater *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Greater>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::GHIJ>::visit(const luci::CircleGreaterEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::GreaterEqual>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::GHIJ>::visit(const luci::CircleIf *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::GHIJ>::visit(const luci::CircleInstanceNorm *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *gamma = getInputTensor(node->gamma());
  const Tensor *beta = getInputTensor(node->beta());

  Tensor *output = getOutputTensor(node);

  InstanceNormParams params{};
  params.epsilon = node->epsilon();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::InstanceNorm>(input, gamma, beta, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::GHIJ>::visit(const luci::CircleInput *)
{
  throw std::runtime_error("Input node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleL2Normalize *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  L2NormParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Normalize>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleL2Pool2D *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLeakyRelu *node)
{
  assert(node->arity() == 1);
  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  LeakyReluParams params{};
  params.alpha = node->alpha();

  return std::make_unique<kernels::LeakyRelu>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLess *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Less>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLessEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LessEqual>(x, y, output);
}

std::unique_ptr<Kernel>
KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLocalResponseNormalization *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLogicalAnd *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalAnd>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLogicalNot *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalNot>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLogicalOr *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalOr>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLogistic *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Logistic>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleLogSoftmax *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogSoftmax>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMaximum *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Maximum>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMaxPool2D *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMean *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *axes = getInputTensor(node->reduction_indices());
  Tensor *output = getOutputTensor(node);

  auto temp_index_unique =
    std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
  temp_index_unique->make_unobservable();
  temp_index_unique->set_data_buffer(nullptr);
  Tensor *temp_index = getRuntimeGraph(node->graph())->addTensor(std::move(temp_index_unique));

  auto resolved_axes_unique =
    std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
  resolved_axes_unique->make_unobservable();
  resolved_axes_unique->set_data_buffer(nullptr);
  Tensor *resolved_axes =
    getRuntimeGraph(node->graph())->addTensor(std::move(resolved_axes_unique));

  auto temp_sum_unique =
    std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  temp_sum_unique->make_unobservable();
  temp_sum_unique->set_data_buffer(nullptr);
  Tensor *temp_sum = getRuntimeGraph(node->graph())->addTensor(std::move(temp_sum_unique));

  ReducerParams params{};
  params.keep_dims = node->keep_dims();

  return std::make_unique<kernels::Mean>(input, axes, output, temp_index, resolved_axes, temp_sum,
                                         params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMinimum *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Minimum>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMirrorPad *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *paddings = getInputTensor(node->paddings());
  Tensor *output = getOutputTensor(node);

  MirrorPadParams params{};
  params.mode = node->mode();

  return std::make_unique<kernels::MirrorPad>(input, paddings, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleMul *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  MulParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Mul>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleNeg *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Neg>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::KLMN>::visit(const luci::CircleNotEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::NotEqual>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleOutput *)
{
  throw std::runtime_error("Output node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CirclePack *node)
{
  assert(node->arity() == node->values_count());

  std::vector<const Tensor *> inputs(node->values_count());
  for (uint32_t i = 0; i < node->values_count(); ++i)
  {
    inputs[i] = getInputTensor(node->values(i));
  }
  Tensor *output = getOutputTensor(node);

  PackParams params{};
  params.axis = node->axis();
  params.values_count = node->values_count();

  return std::make_unique<kernels::Pack>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CirclePad *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *paddings = getInputTensor(node->paddings());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Pad>(input, paddings, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CirclePadV2 *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *paddings = getInputTensor(node->paddings());
  const Tensor *constant_values = getInputTensor(node->constant_values());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::PadV2>(input, paddings, constant_values, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CirclePow *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());

  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Pow>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CirclePRelu *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *alpha = getInputTensor(node->alpha());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::PRelu>(input, alpha, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleRelu *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Relu>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleRelu6 *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Relu6>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleReshape *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *shape = getInputTensor(node->shape());
  Tensor *output = getOutputTensor(node);

  // NOTE 'newShape' attribute is ignored.
  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleResizeBilinear *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *size = getInputTensor(node->size());
  Tensor *output = getOutputTensor(node);

  ResizeBilinearParams params{};
  params.align_corners = node->align_corners();
  params.half_pixel_centers = node->half_pixel_centers();

  return std::make_unique<kernels::ResizeBilinear>(input, size, output, params);
}

std::unique_ptr<Kernel>
KernelBuilderLet<KB::OPQR>::visit(const luci::CircleResizeNearestNeighbor *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *size = getInputTensor(node->size());
  Tensor *output = getOutputTensor(node);

  ResizeNearestNeighborParams params{};
  params.align_corners = node->align_corners();
  // TODO update half_pixel_centers after CircleResizeNearestNeighbor updated
  // Current CircleResizeNearestNeighbor don't have half_pixel_centers.
  // default value on current is false.
  // it need to be updated when CircleResizeNearestNeighbor updated.
  params.half_pixel_centers = false;

  return std::make_unique<kernels::ResizeNearestNeighbor>(input, size, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleReverseV2 *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *axes = getInputTensor(node->axis());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::ReverseV2>(input, axes, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::OPQR>::visit(const luci::CircleRsqrt *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Rsqrt>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSlice *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *begin = getInputTensor(node->begin());
  const Tensor *size = getInputTensor(node->size());

  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Slice>(input, begin, size, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSoftmax *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  SoftmaxParams params{};
  params.beta = node->beta();

  return std::make_unique<kernels::Softmax>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSpaceToBatchND *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *block_shape = getInputTensor(node->block_shape());
  const Tensor *paddings = getInputTensor(node->paddings());

  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::SpaceToBatchND>(input, block_shape, paddings, output);
  ;
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSpaceToDepth *node)
{
  assert(node->arity() == 1);
  const Tensor *input = getInputTensor(node->input());

  Tensor *output = getOutputTensor(node);

  SpaceToDepthParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::SpaceToDepth>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSplit *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleSplitOut>(node);
  assert(node->arity() == 2);
  assert(output_nodes.size() == static_cast<size_t>(node->num_split()));

  const Tensor *axis = getInputTensor(node->split_dim());
  const Tensor *input = getInputTensor(node->input());
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  // NOTE 'num_splits' attribute is ignored.
  return std::make_unique<kernels::Split>(axis, input, std::move(outputs));
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSqrt *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Sqrt>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSquare *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Square>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSquaredDifference *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::SquaredDifference>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSqueeze *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->input());
  Tensor *output = getOutputTensor(node);

  SqueezeParams params{};
  params.squeeze_dims = node->squeeze_dims();

  return std::make_unique<kernels::Squeeze>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleStridedSlice *node)
{
  assert(node->arity() == 4);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *begin = getInputTensor(node->begin());
  const Tensor *end = getInputTensor(node->end());
  const Tensor *strides = getInputTensor(node->strides());

  Tensor *output = getOutputTensor(node);

  StridedSliceParams params{};
  params.begin_mask = node->begin_mask();
  params.ellipsis_mask = node->ellipsis_mask();
  params.end_mask = node->end_mask();
  params.new_axis_mask = node->new_axis_mask();
  params.shrink_axis_mask = node->shrink_axis_mask();

  return std::make_unique<kernels::StridedSlice>(input, begin, end, strides, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleSub *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  SubParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Sub>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleTanh *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Tanh>(input, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleTranspose *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->a());
  const Tensor *perm = getInputTensor(node->perm());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Transpose>(input, perm, output);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleTransposeConv *node)
{
  assert(node->arity() == 4);

  const Tensor *input_sizes = getInputTensor(node->inputSizes());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *out_backprop = getInputTensor(node->outBackprop());
  const Tensor *bias = getOptionalInputTensor(node->bias());

  Tensor *output = getOutputTensor(node);

  DataType scratch_data_type =
    getInputTensor(node)->element_type() == DataType::S16 ? DataType::S64 : DataType::S32;

  auto scratch_tensor =
    std::make_unique<Tensor>(scratch_data_type, Shape({}), AffineQuantization{}, "");
  scratch_tensor->make_unobservable();
  scratch_tensor->set_data_buffer(nullptr);
  Tensor *tmp = getRuntimeGraph(node->graph())->addTensor(std::move(scratch_tensor));

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
                                                  tmp, params);
}

std::unique_ptr<Kernel> KernelBuilderLet<KB::STUV>::visit(const luci::CircleUnpack *node)
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

std::unique_ptr<Kernel> KernelBuilderLet<KB::WXYZ>::visit(const luci::CircleWhile *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleWhileOut>(node);
  assert(node->arity() == node->input_count());
  assert(output_nodes.size() == static_cast<size_t>(node->output_count()));

  std::vector<const Tensor *> inputs(node->input_count());
  for (uint32_t i = 0; i < node->input_count(); ++i)
  {
    inputs[i] = getInputTensor(node->input(i));
  }
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  RuntimeGraph *cond_graph = getRuntimeGraph(node->cond_graph());
  RuntimeGraph *body_graph = getRuntimeGraph(node->body_graph());

  return std::make_unique<kernels::While>(std::move(inputs), std::move(outputs), cond_graph,
                                          body_graph);
}

} // namespace luci_interpreter
