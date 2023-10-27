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

#include "loader/GraphLoader.h"
#include "loader/KernelBuilder.h"
#include "luci_interpreter/SimpleMemoryManager.h"

#include <kernels/Add.h>
#include <kernels/ArgMax.h>
#include <kernels/AveragePool2D.h>
#include <kernels/BatchMatMul.h>
#include <kernels/Cast.h>
#include <kernels/Concatenation.h>
#include <kernels/Conv2D.h>
#include <kernels/Cos.h>
#include <kernels/DepthToSpace.h>
#include <kernels/DepthwiseConv2D.h>
#include <kernels/Div.h>
#include <kernels/Elu.h>
#include <kernels/Exp.h>
#include <kernels/Floor.h>
#include <kernels/FloorDiv.h>
#include <kernels/Equal.h>
#include <kernels/FullyConnected.h>
#include <kernels/Greater.h>
#include <kernels/GreaterEqual.h>
#include <kernels/InstanceNorm.h>
#include <kernels/L2Normalize.h>
#include <kernels/L2Pool2D.h>
#include <kernels/LeakyRelu.h>
#include <kernels/Less.h>
#include <kernels/LessEqual.h>
#include <kernels/LocalResponseNormalization.h>
#include <kernels/LogicalAnd.h>
#include <kernels/LogicalNot.h>
#include <kernels/LogicalOr.h>
#include <kernels/Logistic.h>
#include <kernels/LogSoftmax.h>
#include <kernels/Maximum.h>
#include <kernels/MaxPool2D.h>
#include <kernels/Mean.h>
#include <kernels/Minimum.h>
#include <kernels/Mul.h>
#include <kernels/Neg.h>
#include <kernels/NotEqual.h>
#include <kernels/OneHot.h>
#include <kernels/Pad.h>
#include <kernels/PadV2.h>
#include <kernels/Pow.h>
#include <kernels/PRelu.h>
#include <kernels/Relu.h>
#include <kernels/Relu6.h>
#include <kernels/Reshape.h>
#include <kernels/ResizeBilinear.h>
#include <kernels/ResizeNearestNeighbor.h>
#include <kernels/ReverseV2.h>
#include <kernels/Rsqrt.h>
#include <kernels/Sin.h>
#include <kernels/Slice.h>
#include <kernels/Softmax.h>
#include <kernels/SpaceToDepth.h>
#include <kernels/Split.h>
#include <kernels/SplitV.h>
#include <kernels/Sqrt.h>
#include <kernels/SquaredDifference.h>
#include <kernels/Squeeze.h>
#include <kernels/StridedSlice.h>
#include <kernels/Sub.h>
#include <kernels/Tanh.h>
#include <kernels/Tile.h>
#include <kernels/Transpose.h>
#include <kernels/TransposeConv.h>
#include <kernels/Unpack.h>

#include <gmock/gmock.h>

namespace luci_interpreter
{
namespace
{

using namespace testing;

class KernelBuilderTest : public Test
{
protected:
  luci::CircleInput *createInputNode() { return createNode<luci::CircleInput>(); }
  void SetUp() override { _memory_manager = std::make_unique<SimpleMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;

  template <typename NodeT, typename... Args> NodeT *createNode(Args &&... args)
  {
    auto *node = _graph.nodes()->create<NodeT>(std::forward<Args>(args)...);
    // The actual type does not matter for the purpose of the tests.
    // NOTE The type is meaningless for nodes with multiple outputs (corresponding *Out nodes carry
    //  actual output types).
    node->dtype(loco::DataType::FLOAT32);
    return node;
  }

  template <typename NodeOutT> NodeOutT *createNodeOut(loco::Node *node, int index)
  {
    auto *node_out = createNode<NodeOutT>();
    node_out->input(node);
    node_out->index(index);
    return node_out;
  }

  template <typename KernelT> std::unique_ptr<KernelT> buildKernel(const luci::CircleNode *op)
  {
    std::unordered_map<const loco::Graph *, RuntimeGraph *> graph_to_runtime_graph;

    RuntimeGraph runtime_graph(nullptr, _memory_manager.get());
    graph_to_runtime_graph[&_graph] = &runtime_graph;
    RuntimeToIR runtime_to_ir;
    GraphLoader graph_loader(&_graph, &runtime_graph, runtime_to_ir, graph_to_runtime_graph,
                             _node_to_tensor, _memory_manager.get());
    graph_loader.loadTensors();

    KernelBuilder kernel_builder(graph_to_runtime_graph, _node_to_tensor);

    auto kernel = kernel_builder.build(op);
    return std::unique_ptr<KernelT>(dynamic_cast<KernelT *>(kernel.release()));
  }

  void checkTensor(const Tensor *tensor, const loco::Node *node)
  {
    EXPECT_THAT(tensor, Eq(_node_to_tensor.at(node)));
  }

private:
  loco::Graph _graph;
  std::unordered_map<const loco::Node *, Tensor *> _node_to_tensor;
};

TEST_F(KernelBuilderTest, Add)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleAdd>();
  op->x(input1);
  op->y(input2);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::Add>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, ArgMax)
{
  auto *input = createInputNode();
  auto *axis = createInputNode();

  auto *op = createNode<luci::CircleArgMax>();
  op->input(input);
  op->dimension(axis);

  op->output_type(loco::DataType::FLOAT32);

  auto kernel = buildKernel<kernels::ArgMax>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->axis(), axis);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().output_type, Eq(op->output_type()));
}

TEST_F(KernelBuilderTest, AveragePool2D)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleAveragePool2D>();
  op->value(input);

  op->padding(luci::Padding::SAME);
  op->filter()->h(11);
  op->filter()->w(13);
  op->stride()->h(17);
  op->stride()->w(19);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::AveragePool2D>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().filter_height, Eq(op->filter()->h()));
  EXPECT_THAT(kernel->params().filter_width, Eq(op->filter()->w()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, BatchMatMul)
{
  auto *lhs = createInputNode();
  auto *rhs = createInputNode();

  auto *op = createNode<luci::CircleBatchMatMul>();
  op->x(lhs);
  op->y(rhs);
  op->adj_x(false);
  op->adj_y(false);

  auto kernel = buildKernel<kernels::BatchMatMul>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), lhs);
  checkTensor(kernel->y(), rhs);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().adj_x, Eq(op->adj_x()));
  EXPECT_THAT(kernel->params().adj_y, Eq(op->adj_y()));
}

TEST_F(KernelBuilderTest, Cast)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleCast>();
  op->x(input);

  auto kernel = buildKernel<kernels::Cast>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Concatenation)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleConcatenation>(2);
  op->values(0, input1);
  op->values(1, input2);
  op->axis(11);

  auto kernel = buildKernel<kernels::Concatenation>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(0), input1);
  checkTensor(kernel->input(1), input2);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().axis, Eq(op->axis()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Conv2D)
{
  auto *input = createInputNode();
  auto *filter = createInputNode();
  auto *bias = createInputNode();

  auto *op = createNode<luci::CircleConv2D>();
  op->input(input);
  op->filter(filter);
  op->bias(bias);

  op->padding(luci::Padding::SAME);
  op->stride()->h(11);
  op->stride()->w(13);
  op->dilation()->h(17);
  op->dilation()->w(19);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::Conv2D>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->filter(), filter);
  checkTensor(kernel->bias(), bias);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().dilation_height_factor, Eq(op->dilation()->h()));
  EXPECT_THAT(kernel->params().dilation_width_factor, Eq(op->dilation()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Cos)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleCos>();
  op->x(input);

  auto kernel = buildKernel<kernels::Cos>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, DepthToSpace)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleDepthToSpace>();
  op->input(input);

  op->block_size(11);

  auto kernel = buildKernel<kernels::DepthToSpace>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().block_size, Eq(op->block_size()));
}

TEST_F(KernelBuilderTest, DepthwiseConv2D)
{
  auto *input = createInputNode();
  auto *filter = createInputNode();
  auto *bias = createInputNode();

  auto *op = createNode<luci::CircleDepthwiseConv2D>();
  op->input(input);
  op->filter(filter);
  op->bias(bias);

  op->padding(luci::Padding::SAME);
  op->depthMultiplier(11);
  op->stride()->h(13);
  op->stride()->w(17);
  op->dilation()->h(19);
  op->dilation()->w(23);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::DepthwiseConv2D>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->filter(), filter);
  checkTensor(kernel->bias(), bias);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().depth_multiplier, Eq(op->depthMultiplier()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().dilation_height_factor, Eq(op->dilation()->h()));
  EXPECT_THAT(kernel->params().dilation_width_factor, Eq(op->dilation()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Div)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleDiv>();
  op->x(input1);
  op->y(input2);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::Div>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Elu)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleElu>();
  op->features(input);

  auto kernel = buildKernel<kernels::Elu>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Exp)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleExp>();
  op->x(input);

  auto kernel = buildKernel<kernels::Exp>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Floor)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleFloor>();
  op->x(input);

  auto kernel = buildKernel<kernels::Floor>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, FloorDiv)
{
  auto *x = createInputNode();
  auto *y = createInputNode();

  auto *op = createNode<luci::CircleFloorDiv>();
  op->x(x);
  op->y(y);

  auto kernel = buildKernel<kernels::FloorDiv>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x);
  checkTensor(kernel->y(), y);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Equal)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleEqual>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::Equal>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, FullyConnected)
{
  auto *input = createInputNode();
  auto *weights = createInputNode();
  auto *bias = createInputNode();

  auto *op = createNode<luci::CircleFullyConnected>();
  op->input(input);
  op->weights(weights);
  op->bias(bias);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::FullyConnected>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->weights(), weights);
  checkTensor(kernel->bias(), bias);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Greater)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleGreater>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::Greater>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, GreaterEqual)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleGreaterEqual>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::GreaterEqual>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, InstanceNorm)
{
  auto *input = createInputNode();
  auto *gamma = createInputNode();
  auto *beta = createInputNode();

  auto *op = createNode<luci::CircleInstanceNorm>();
  op->input(input);
  op->gamma(gamma);
  op->beta(beta);

  op->epsilon(1e-05);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::InstanceNorm>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->gamma(), gamma);
  checkTensor(kernel->beta(), beta);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().epsilon, Eq(op->epsilon()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, L2Normalize)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleL2Normalize>();
  op->x(input);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::L2Normalize>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, L2Pool2D)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleL2Pool2D>();
  op->value(input);

  op->padding(luci::Padding::SAME);
  op->filter()->h(11);
  op->filter()->w(13);
  op->stride()->h(17);
  op->stride()->w(19);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::L2Pool2D>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().filter_height, Eq(op->filter()->h()));
  EXPECT_THAT(kernel->params().filter_width, Eq(op->filter()->w()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, LeakyRelu)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleLeakyRelu>();
  op->features(input);

  op->alpha(11.0f);

  auto kernel = buildKernel<kernels::LeakyRelu>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().alpha, Eq(op->alpha()));
}

TEST_F(KernelBuilderTest, Less)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleLess>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::Less>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, LessEqual)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleLessEqual>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::LessEqual>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, LocalResponseNormalization)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleLocalResponseNormalization>();
  op->input(input);

  op->radius(11);
  op->bias(13.0f);
  op->alpha(15.0f);
  op->beta(17.0f);

  auto kernel = buildKernel<kernels::LocalResponseNormalization>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().radius, Eq(op->radius()));
  EXPECT_THAT(kernel->params().bias, Eq(op->bias()));
  EXPECT_THAT(kernel->params().alpha, Eq(op->alpha()));
  EXPECT_THAT(kernel->params().beta, Eq(op->beta()));
}

TEST_F(KernelBuilderTest, LogicalAnd)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleLogicalAnd>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::LogicalAnd>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, LogicalNot)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleLogicalNot>();
  op->x(input);

  auto kernel = buildKernel<kernels::LogicalNot>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, LogicalOr)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleLogicalOr>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::LogicalOr>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Logistic)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleLogistic>();
  op->x(input);

  auto kernel = buildKernel<kernels::Logistic>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, LogSoftmax)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleLogSoftmax>();
  op->logits(input);

  auto kernel = buildKernel<kernels::LogSoftmax>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Maximum)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleMaximum>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::Maximum>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, MaxPool2D)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleMaxPool2D>();
  op->value(input);

  op->padding(luci::Padding::SAME);
  op->filter()->h(11);
  op->filter()->w(13);
  op->stride()->h(17);
  op->stride()->w(19);
  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::MaxPool2D>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().filter_height, Eq(op->filter()->h()));
  EXPECT_THAT(kernel->params().filter_width, Eq(op->filter()->w()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Mean)
{
  auto *input = createInputNode();
  auto *axes = createInputNode();

  auto *op = createNode<luci::CircleMean>();
  op->input(input);
  op->reduction_indices(axes);

  op->keep_dims(true);

  auto kernel = buildKernel<kernels::Mean>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->axes(), axes);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().keep_dims, Eq(op->keep_dims()));
}

TEST_F(KernelBuilderTest, Minimum)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleMinimum>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::Minimum>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Mul)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleMul>();
  op->x(input1);
  op->y(input2);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::Mul>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Neg)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleNeg>();
  op->x(input);

  auto kernel = buildKernel<kernels::Neg>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, NotEqual)
{
  auto *x_input = createInputNode();
  auto *y_input = createInputNode();

  auto *op = createNode<luci::CircleNotEqual>();
  op->x(x_input);
  op->y(y_input);

  auto kernel = buildKernel<kernels::NotEqual>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->x(), x_input);
  checkTensor(kernel->y(), y_input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, OneHot)
{
  auto *indices = createInputNode();
  auto *depth = createInputNode();
  auto *on_value = createInputNode();
  auto *off_value = createInputNode();
  auto axis = 1;

  auto *op = createNode<luci::CircleOneHot>();
  op->indices(indices);
  op->depth(depth);
  op->on_value(on_value);
  op->off_value(off_value);
  op->axis(axis);

  auto kernel = buildKernel<kernels::OneHot>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->indices(), indices);
  checkTensor(kernel->depth(), depth);
  checkTensor(kernel->on_value(), on_value);
  checkTensor(kernel->off_value(), off_value);
  EXPECT_THAT(kernel->params().axis, Eq(op->axis()));
}

TEST_F(KernelBuilderTest, Pad)
{
  auto *input = createInputNode();
  auto *paddings = createInputNode();

  auto *op = createNode<luci::CirclePad>();
  op->input(input);
  op->paddings(paddings);

  auto kernel = buildKernel<kernels::Pad>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->paddings(), paddings);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, PadV2)
{
  auto *input = createInputNode();
  auto *paddings = createInputNode();
  auto *constant_values = createInputNode();

  auto *op = createNode<luci::CirclePadV2>();
  op->input(input);
  op->paddings(paddings);
  op->constant_values(constant_values);

  auto kernel = buildKernel<kernels::PadV2>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->paddings(), paddings);
  checkTensor(kernel->constant_values(), constant_values);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Pow)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CirclePow>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::Pow>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, PRelu)
{
  auto *input = createInputNode();
  auto *alpha = createInputNode();

  auto *op = createNode<luci::CirclePRelu>();
  op->input(input);
  op->alpha(alpha);

  auto kernel = buildKernel<kernels::PRelu>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->alpha(), alpha);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Relu)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleRelu>();
  op->features(input);

  auto kernel = buildKernel<kernels::Relu>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Relu6)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleRelu6>();
  op->features(input);

  auto kernel = buildKernel<kernels::Relu6>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Reshape)
{
  auto *input = createInputNode();
  auto *shape = createInputNode();

  auto *op = createNode<luci::CircleReshape>();
  op->tensor(input);
  op->shape(shape);

  auto kernel = buildKernel<kernels::Reshape>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->shape(), shape);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, ResizeBilinear)
{
  auto *input = createInputNode();
  auto *size = createInputNode();

  auto *op = createNode<luci::CircleResizeBilinear>();
  op->input(input);
  op->size(size);
  op->align_corners(true);
  op->half_pixel_centers(true);

  auto kernel = buildKernel<kernels::ResizeBilinear>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->size(), size);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().align_corners, Eq(op->align_corners()));
  EXPECT_THAT(kernel->params().half_pixel_centers, Eq(op->half_pixel_centers()));
}

TEST_F(KernelBuilderTest, ResizeNearestNeighbor)
{
  auto *input = createInputNode();
  auto *size = createInputNode();

  auto *op = createNode<luci::CircleResizeNearestNeighbor>();
  op->input(input);
  op->size(size);
  op->align_corners(true);

  auto kernel = buildKernel<kernels::ResizeNearestNeighbor>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->size(), size);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().align_corners, Eq(op->align_corners()));
  // TODO currently half_pixel_centers are not implemented on CircleResizeNearestNeighbor
  // after adding, need to be updated.
}

TEST_F(KernelBuilderTest, ReverseV2)
{
  auto *input = createInputNode();
  auto *axes = createInputNode();

  auto *op = createNode<luci::CircleReverseV2>();
  op->tensor(input);
  op->axis(axes);

  auto kernel = buildKernel<kernels::ReverseV2>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->axes(), axes);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Rsqrt)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleRsqrt>();
  op->x(input);

  auto kernel = buildKernel<kernels::Rsqrt>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Sin)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleSin>();
  op->x(input);

  auto kernel = buildKernel<kernels::Sin>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Slice)
{
  auto *input = createInputNode();
  auto *begin = createInputNode();
  auto *size = createInputNode();

  auto *op = createNode<luci::CircleSlice>();
  op->input(input);
  op->begin(begin);
  op->size(size);

  auto kernel = buildKernel<kernels::Slice>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->begin(), begin);
  checkTensor(kernel->size(), size);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Softmax)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleSoftmax>();
  op->logits(input);

  op->beta(11.0f);

  auto kernel = buildKernel<kernels::Softmax>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().beta, Eq(op->beta()));
}

TEST_F(KernelBuilderTest, SpaceToDepth)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleSpaceToDepth>();
  op->input(input);

  op->block_size(11);

  auto kernel = buildKernel<kernels::SpaceToDepth>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().block_size, op->block_size());
}

TEST_F(KernelBuilderTest, Split)
{
  auto *axis = createInputNode();
  auto *input = createInputNode();
  auto *op = createNode<luci::CircleSplit>();
  auto *output1 = createNodeOut<luci::CircleSplitOut>(op, 0);
  auto *output2 = createNodeOut<luci::CircleSplitOut>(op, 1);

  op->split_dim(axis);
  op->input(input);

  op->num_split(2);

  auto kernel = buildKernel<kernels::Split>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->axis(), axis);
  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(0), output1);
  checkTensor(kernel->output(1), output2);
}

TEST_F(KernelBuilderTest, SplitV)
{
  auto *input = createInputNode();
  auto *size_splits = createInputNode();
  auto *axis = createInputNode();
  auto *op = createNode<luci::CircleSplitV>();
  auto *output0 = createNodeOut<luci::CircleSplitVOut>(op, 0);
  auto *output1 = createNodeOut<luci::CircleSplitVOut>(op, 1);

  op->input(input);
  op->size_splits(size_splits);
  op->split_dim(axis);

  op->num_split(2);

  auto kernel = buildKernel<kernels::SplitV>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->size_splits(), size_splits);
  checkTensor(kernel->axis(), axis);
  checkTensor(kernel->output(0), output0);
  checkTensor(kernel->output(1), output1);
}

TEST_F(KernelBuilderTest, Sqrt)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleSqrt>();
  op->x(input);

  auto kernel = buildKernel<kernels::Sqrt>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, SquaredDifference)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleSquaredDifference>();
  op->x(input1);
  op->y(input2);

  auto kernel = buildKernel<kernels::SquaredDifference>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Squeeze)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleSqueeze>();
  op->input(input);

  op->squeeze_dims({11, 13});

  auto kernel = buildKernel<kernels::Squeeze>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().squeeze_dims, ElementsAreArray(op->squeeze_dims()));
}

TEST_F(KernelBuilderTest, StridedSlice)
{
  auto *input = createInputNode();
  auto *begin = createInputNode();
  auto *end = createInputNode();
  auto *strides = createInputNode();

  auto *op = createNode<luci::CircleStridedSlice>();
  op->input(input);
  op->begin(begin);
  op->end(end);
  op->strides(strides);

  op->begin_mask(11);
  op->ellipsis_mask(13);
  op->end_mask(17);
  op->new_axis_mask(19);
  op->shrink_axis_mask(23);

  auto kernel = buildKernel<kernels::StridedSlice>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->begin(), begin);
  checkTensor(kernel->end(), end);
  checkTensor(kernel->strides(), strides);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().begin_mask, Eq(op->begin_mask()));
  EXPECT_THAT(kernel->params().ellipsis_mask, Eq(op->ellipsis_mask()));
  EXPECT_THAT(kernel->params().end_mask, Eq(op->end_mask()));
  EXPECT_THAT(kernel->params().new_axis_mask, Eq(op->new_axis_mask()));
  EXPECT_THAT(kernel->params().shrink_axis_mask, Eq(op->shrink_axis_mask()));
}

TEST_F(KernelBuilderTest, Sub)
{
  auto *input1 = createInputNode();
  auto *input2 = createInputNode();

  auto *op = createNode<luci::CircleSub>();
  op->x(input1);
  op->y(input2);

  op->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto kernel = buildKernel<kernels::Sub>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input1(), input1);
  checkTensor(kernel->input2(), input2);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Tanh)
{
  auto *input = createInputNode();

  auto *op = createNode<luci::CircleTanh>();
  op->x(input);

  auto kernel = buildKernel<kernels::Tanh>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Tile)
{
  auto *input = createInputNode();
  auto *multiples = createInputNode();

  auto *op = createNode<luci::CircleTile>();
  op->input(input);
  op->multiples(multiples);

  auto kernel = buildKernel<kernels::Tile>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->multiples(), multiples);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, Transpose)
{
  auto *input = createInputNode();
  auto *perm = createInputNode();

  auto *op = createNode<luci::CircleTranspose>();
  op->a(input);
  op->perm(perm);

  auto kernel = buildKernel<kernels::Transpose>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->perm(), perm);
  checkTensor(kernel->output(), op);
}

TEST_F(KernelBuilderTest, TransposeConv)
{
  auto *output_shape = createInputNode();
  auto *filter = createInputNode();
  auto *input = createInputNode();
  auto *bias = createInputNode();

  auto *op = createNode<luci::CircleTransposeConv>();
  op->inputSizes(output_shape);
  op->filter(filter);
  op->outBackprop(input);
  op->bias(bias);

  op->padding(luci::Padding::SAME);
  op->stride()->h(11);
  op->stride()->w(13);
  op->fusedActivationFunction(luci::FusedActFunc::NONE);

  auto kernel = buildKernel<kernels::TransposeConv>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->output_shape(), output_shape);
  checkTensor(kernel->filter(), filter);
  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  checkTensor(kernel->bias(), bias);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
  EXPECT_THAT(kernel->params().activation, Eq(op->fusedActivationFunction()));
}

TEST_F(KernelBuilderTest, Unpack)
{
  auto *input = createInputNode();
  auto *op = createNode<luci::CircleUnpack>();
  auto *output1 = createNodeOut<luci::CircleUnpackOut>(op, 0);
  auto *output2 = createNodeOut<luci::CircleUnpackOut>(op, 1);

  op->value(input);

  op->num(2);
  op->axis(11);

  auto kernel = buildKernel<kernels::Unpack>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(0), output1);
  checkTensor(kernel->output(1), output2);
  EXPECT_THAT(kernel->params().axis, Eq(op->axis()));
}

TEST_F(KernelBuilderTest, NonExisting1_NEG)
{
  auto *op = createNode<luci::CircleConst>();
  ASSERT_ANY_THROW(buildKernel<Kernel>(op));
}

TEST_F(KernelBuilderTest, NonExisting2_NEG)
{
  auto *op = createNode<luci::CircleInput>();
  ASSERT_ANY_THROW(buildKernel<Kernel>(op));
}

TEST_F(KernelBuilderTest, NonExisting3_NEG)
{
  auto *op = createNode<luci::CircleOutput>();
  ASSERT_ANY_THROW(buildKernel<Kernel>(op));
}

} // namespace
} // namespace luci_interpreter
