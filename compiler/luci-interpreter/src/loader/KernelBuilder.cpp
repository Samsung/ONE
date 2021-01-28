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
#include "kernels/Compiled.h"
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
#include "kernels/Mul.h"
#include "kernels/Neg.h"
#include "kernels/NotEqual.h"
#include "kernels/Pad.h"
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
#include "kernels/SpaceToDepth.h"
#include "kernels/Split.h"
#include "kernels/StridedSlice.h"
#include "kernels/Sqrt.h"
#include "kernels/Sub.h"
#include "kernels/Squeeze.h"
#include "kernels/Tanh.h"
#include "kernels/Unpack.h"
#include "kernels/Transpose.h"
#include "kernels/TransposeConv.h"

#include "Halide.h"
#include "flatbuffers/flexbuffers.h"

#include <stdexcept>

#include <dlfcn.h>

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

static Halide::Type to_halide_type(loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      return Halide::Type(Halide::Type::Float, 32, 1);
    case loco::DataType::FLOAT64:
      return Halide::Type(Halide::Type::Float, 64, 1);
    case loco::DataType::S32:
      return Halide::Type(Halide::Type::Int, 32, 1);
    case loco::DataType::S64:
      return Halide::Type(Halide::Type::Int, 64, 1);
    default:
      assert("NYI");
  }
  return Halide::Type();
}

const Tensor *KernelBuilder::getInputTensor(const loco::Node *node) const
{
  const Tensor *tensor = _node_to_tensor.at(node);
  assert(tensor != nullptr);
  return tensor;
}

const Tensor *KernelBuilder::getOptionalInputTensor(const loco::Node *node) const
{
  if (dynamic_cast<const luci::CircleOutputExclude *>(node))
  {
    return nullptr;
  }
  return getInputTensor(node);
}

Tensor *KernelBuilder::getOutputTensor(const loco::Node *node) const
{
  Tensor *tensor = _node_to_tensor.at(node);
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
  RuntimeGraph *runtime_graph = _graph_to_runtime_graph.at(graph);
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
  const Tensor *input = getInputTensor(node->input());
  const Tensor *axis = getInputTensor(node->dimension());
  Tensor *output = getOutputTensor(node);

  ArgMaxParams params{};
  params.output_type = node->output_type();

  return std::make_unique<kernels::ArgMax>(input, axis, output, params);
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
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Concatenation>(std::move(inputs), output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConst *)
{
  throw std::runtime_error("Const node cannot be executed.");
}

static void add_argument_halide_buffer(std::vector<Halide::Runtime::Buffer<>> &arguments, const loco::Node *node)
{
  const luci::CircleNode *luci_node = static_cast<const luci::CircleNode *>(node);
  std::vector<int> dims;
  for (int j = 0; j < luci_node->arity(); ++j)
  {
    assert(luci_node->dim(j).known());
    dims.push_back(luci_node->dim(j).value()); // TODO check if dims order is correct
  }
  arguments.emplace_back(to_halide_type(luci_node->dtype()), nullptr, dims);
  assert(!arguments.back().owns_host_memory());
}

static auto get_custom_function_impl(const luci::CircleCustom *node) -> void (*)(void **)
{
  // get function name
  const auto &options_buffer = node->custom_options();
  auto options_map = flexbuffers::GetRoot(options_buffer).AsMap();
  auto func_name = options_map["func_name"].AsString().str() + "_argv";

  // get pointer from name
  auto handle = dlopen("libcompiled.so", RTLD_LAZY);
  if (!handle)
  {
    std::cerr << dlerror();
    exit(EXIT_FAILURE);
  }

  auto actual_func = reinterpret_cast<void (*)(void **)>(dlsym(handle, func_name.c_str()));

  return actual_func;
}

static std::vector<luci_interpreter::Shape> get_custom_output_shapes(const std::vector<const loco::Node *> &nodes)
{
  std::vector<luci_interpreter::Shape> shapes;
  for (int i = 0; i < nodes.size(); ++i)
  {
    auto out_node = static_cast<const luci::CircleCustomOut *>(nodes[i]);
    assert(out_node->index() == i);
    luci_interpreter::Shape out_shape(out_node->rank());
    for (int j = 0; j < out_node->rank(); ++j)
    {
      assert(out_node->dim(j).known());
      out_shape.dim(j) = out_node->dim(j).value();
    }
    shapes.emplace_back(std::move(out_shape));
  }
  return shapes;
}

static CompiledParams::OperationImpl get_custom_function(const luci::CircleCustom *node)
{
  std::vector<Halide::Runtime::Buffer<>> arguments;

  std::vector<const loco::Node *> output_nodes = collectOutputNodes<luci::CircleCustomOut>(node);

  for (int i = 0; i < node->numInputs(); ++i)
  {
    add_argument_halide_buffer(arguments, node->inputs(i));
  }

  for (int i = 0; i < output_nodes.size(); ++i)
  {
    add_argument_halide_buffer(arguments, output_nodes[i]);
  }

  auto actual_func = get_custom_function_impl(node);

  auto impl = [arguments, actual_func](std::vector<const char *> inputs, const std::vector<char *> outputs) mutable
  {
    std::vector<void *> raw_arguments; // TODO allocate this on stack, maybe?
    int num_inputs = inputs.size();
    int num_outputs = outputs.size();
    for (int i = 0; i < num_inputs; ++i)
    {
      halide_buffer_t *buf = arguments[i].raw_buffer();
      buf->host = reinterpret_cast<uint8_t*>(const_cast<char *>(inputs[i]));  // a little dirty, but we need this to pass input data to halide generated function
      raw_arguments.push_back(reinterpret_cast<void *>(buf));
    }
    for (int i = 0; i < num_outputs; ++i)
    {
      halide_buffer_t *buf = arguments[num_inputs + i].raw_buffer();
      buf->host = reinterpret_cast<uint8_t*>(outputs[i]);
      raw_arguments.push_back(reinterpret_cast<void *>(buf));
    }
    actual_func(raw_arguments.data());
  };

  return impl;
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleCustom *node)
{
  std::vector<const Tensor *> inputs(node->numInputs());
  for (uint32_t i = 0; i < node->numInputs(); ++i)
  {
    inputs[i] = getInputTensor(node->inputs(i));
  }
  std::vector<const loco::Node *> output_nodes = collectOutputNodes<luci::CircleCustomOut>(node);
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  CompiledParams params{get_custom_output_shapes(output_nodes), get_custom_function(node)};

  return std::make_unique<kernels::Compiled>(std::move(inputs), std::move(outputs), params);
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleDepthToSpace *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->input());
  Tensor *output = getOutputTensor(node);

  DepthToSpaceParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::DepthToSpace>(input, output, params);
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
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::DepthwiseConv2D>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleDiv *node)
{
  assert(node->arity() == 2);
  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  DivParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Div>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleElu *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Elu>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleExp *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Exp>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFloor *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Floor>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFloorDiv *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::FloorDiv>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Equal>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFullyConnected *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleGreater *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Greater>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleGreaterEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::GreaterEqual>(x, y, output);
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleInstanceNorm *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleInput *)
{
  throw std::runtime_error("Input node cannot be executed.");
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLess *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Less>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLessEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LessEqual>(x, y, output);
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogicalAnd *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalAnd>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogicalNot *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalNot>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogicalOr *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogicalOr>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogistic *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Logistic>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleLogSoftmax *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::LogSoftmax>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMaximum *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Maximum>(input1, input2, output);
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

  const Tensor *input = getInputTensor(node->input());
  const Tensor *axes = getInputTensor(node->reduction_indices());
  Tensor *output = getOutputTensor(node);

  ReducerParams params{};
  params.keep_dims = node->keep_dims();

  return std::make_unique<kernels::Mean>(input, axes, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleMinimum *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Minimum>(input1, input2, output);
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleNeg *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Neg>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleNotEqual *node)
{
  assert(node->arity() == 2);

  const Tensor *x = getInputTensor(node->x());
  const Tensor *y = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::NotEqual>(x, y, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleOutput *)
{
  throw std::runtime_error("Output node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CirclePad *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *paddings = getInputTensor(node->paddings());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Pad>(input, paddings, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CirclePow *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());

  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Pow>(input1, input2, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CirclePRelu *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *alpha = getInputTensor(node->alpha());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Prelu>(input, alpha, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleRelu *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Relu>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleRelu6 *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->features());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Relu6>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleReshape *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *shape = getInputTensor(node->shape());
  Tensor *output = getOutputTensor(node);

  // NOTE 'newShape' attribute is ignored.
  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleResizeBilinear *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleResizeNearestNeighbor *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleReverseV2 *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->tensor());
  const Tensor *axes = getInputTensor(node->axis());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Reverse>(input, axes, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleRsqrt *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Rsqrt>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSub *node)
{
  assert(node->arity() == 2);

  const Tensor *input1 = getInputTensor(node->x());
  const Tensor *input2 = getInputTensor(node->y());
  Tensor *output = getOutputTensor(node);

  SubParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Sub>(input1, input2, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSlice *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *begin = getInputTensor(node->begin());
  const Tensor *size = getInputTensor(node->size());

  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Slice>(input, begin, size, output);
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSpaceToDepth *node)
{
  assert(node->arity() == 1);
  const Tensor *input = getInputTensor(node->input());

  Tensor *output = getOutputTensor(node);

  SpaceToDepthParams params{};
  params.block_size = node->block_size();

  return std::make_unique<kernels::SpaceToDepth>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSplit *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSqrt *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Sqrt>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSqueeze *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->input());
  Tensor *output = getOutputTensor(node);

  SqueezeParams params{};
  params.squeeze_dims = node->squeeze_dims();

  return std::make_unique<kernels::Squeeze>(input, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleStridedSlice *node)
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

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleTanh *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->x());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Tanh>(input, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleTranspose *node)
{
  assert(node->arity() == 2);

  const Tensor *input = getInputTensor(node->a());
  const Tensor *perm = getInputTensor(node->perm());
  Tensor *output = getOutputTensor(node);

  return std::make_unique<kernels::Transpose>(input, perm, output);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleTransposeConv *node)
{
  assert(node->arity() == 4);

  const Tensor *input_sizes = getInputTensor(node->inputSizes());
  const Tensor *filter = getInputTensor(node->filter());
  const Tensor *out_backprop = getInputTensor(node->outBackprop());
  const Tensor *bias = getOptionalInputTensor(node->bias());

  Tensor *output = getOutputTensor(node);

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
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

} // namespace luci_interpreter
