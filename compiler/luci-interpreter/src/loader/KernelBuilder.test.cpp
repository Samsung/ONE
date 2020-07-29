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

#include <kernels/Add.h>
#include <kernels/ArgMax.h>
#include <kernels/AveragePool2D.h>
#include <kernels/Concatenation.h>
#include <kernels/Conv2D.h>
#include <kernels/DepthToSpace.h>
#include <kernels/DepthwiseConv2D.h>
#include <kernels/Elu.h>
#include <kernels/FullyConnected.h>
#include <kernels/L2Normalize.h>
#include <kernels/L2Pool2D.h>
#include <kernels/LeakyRelu.h>
#include <kernels/LocalResponseNormalization.h>
#include <kernels/Logistic.h>
#include <kernels/MaxPool2D.h>
#include <kernels/Mean.h>
#include <kernels/Mul.h>
#include <kernels/Pad.h>
#include <kernels/Reshape.h>
#include <kernels/Reverse.h>
#include <kernels/Slice.h>
#include <kernels/Softmax.h>
#include <kernels/SpaceToDepth.h>
#include <kernels/Split.h>
#include <kernels/Squeeze.h>
#include <kernels/StridedSlice.h>
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

    RuntimeGraph runtime_graph(nullptr);
    RuntimeToIR runtime_to_ir;
    GraphLoader graph_loader(&_graph, &runtime_graph, runtime_to_ir, graph_to_runtime_graph,
                             _node_to_tensor);
    graph_loader.loadTensors();

    KernelBuilder kernel_builder(graph_to_runtime_graph, _node_to_tensor);

    auto kernel = op->accept(&kernel_builder);
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

TEST_F(KernelBuilderTest, ReverseV2)
{
  auto *input = createInputNode();
  auto *axes = createInputNode();

  auto *op = createNode<luci::CircleReverseV2>();
  op->tensor(input);
  op->axis(axes);

  auto kernel = buildKernel<kernels::Reverse>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->axes(), axes);
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

  auto *op = createNode<luci::CircleTransposeConv>();
  op->inputSizes(output_shape);
  op->filter(filter);
  op->outBackprop(input);

  op->padding(luci::Padding::SAME);
  op->stride()->h(11);
  op->stride()->w(13);

  auto kernel = buildKernel<kernels::TransposeConv>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->output_shape(), output_shape);
  checkTensor(kernel->filter(), filter);
  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().padding, Eq(op->padding()));
  EXPECT_THAT(kernel->params().stride_height, Eq(op->stride()->h()));
  EXPECT_THAT(kernel->params().stride_width, Eq(op->stride()->w()));
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
