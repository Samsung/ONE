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
#include <kernels/AveragePool2D.h>
#include <kernels/Conv2D.h>
#include <kernels/DepthwiseConv2D.h>
#include <kernels/L2Normalize.h>
#include <kernels/L2Pool2D.h>
#include <kernels/LocalResponseNormalization.h>
#include <kernels/MaxPool2D.h>
#include <kernels/Mul.h>
#include <kernels/Softmax.h>
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
  luci::CircleInput *createInputNode(loco::DataType dtype)
  {
    return createNode<luci::CircleInput>(dtype);
  }

  template <typename NodeT> NodeT *createNode(loco::DataType output_type)
  {
    auto *node = _graph.nodes()->create<NodeT>();
    node->dtype(output_type);
    return node;
  }

  template <typename NodeOutT, typename... DT>
  std::vector<NodeOutT *> createNodeOuts(loco::Node *node, DT... output_types)
  {
    std::vector<NodeOutT *> outputs{createNode<NodeOutT>(output_types)...};
    for (size_t i = 0; i < sizeof...(output_types); ++i)
    {
      outputs[i]->input(node);
      outputs[i]->index(i);
    }
    return outputs;
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
  auto *input1 = createInputNode(loco::DataType::FLOAT32);
  auto *input2 = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleAdd>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, AveragePool2D)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleAveragePool2D>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, Conv2D)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);
  auto *filter = createInputNode(loco::DataType::FLOAT32);
  auto *bias = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleConv2D>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, DepthwiseConv2D)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);
  auto *filter = createInputNode(loco::DataType::FLOAT32);
  auto *bias = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleDepthwiseConv2D>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, L2Normalize)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleL2Normalize>(loco::DataType::FLOAT32);
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
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleL2Pool2D>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, LocalResponseNormalization)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleLocalResponseNormalization>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, MaxPool2D)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleMaxPool2D>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, Mul)
{
  auto *input1 = createInputNode(loco::DataType::FLOAT32);
  auto *input2 = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleMul>(loco::DataType::FLOAT32);
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

TEST_F(KernelBuilderTest, Softmax)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleSoftmax>(loco::DataType::FLOAT32);
  op->logits(input);

  op->beta(11.0f);

  auto kernel = buildKernel<kernels::Softmax>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(), op);
  EXPECT_THAT(kernel->params().beta, Eq(op->beta()));
}

TEST_F(KernelBuilderTest, Transpose)
{
  auto *input = createInputNode(loco::DataType::FLOAT32);
  auto *perm = createInputNode(loco::DataType::S32);

  auto *op = createNode<luci::CircleTranspose>(loco::DataType::FLOAT32);
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
  auto *output_shape = createInputNode(loco::DataType::S32);
  auto *filter = createInputNode(loco::DataType::FLOAT32);
  auto *input = createInputNode(loco::DataType::FLOAT32);

  auto *op = createNode<luci::CircleTransposeConv>(loco::DataType::FLOAT32);
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
  auto *input = createInputNode(loco::DataType::FLOAT32);
  auto *op = createNode<luci::CircleUnpack>(loco::DataType::FLOAT32);
  auto outputs =
      createNodeOuts<luci::CircleUnpackOut>(op, loco::DataType::FLOAT32, loco::DataType::FLOAT32);

  op->value(input);

  op->num(outputs.size());
  op->axis(11);

  auto kernel = buildKernel<kernels::Unpack>(op);
  ASSERT_THAT(kernel, NotNull());

  checkTensor(kernel->input(), input);
  checkTensor(kernel->output(0), outputs[0]);
  checkTensor(kernel->output(1), outputs[1]);
  EXPECT_THAT(kernel->params().axis, Eq(op->axis()));
}

} // namespace
} // namespace luci_interpreter
