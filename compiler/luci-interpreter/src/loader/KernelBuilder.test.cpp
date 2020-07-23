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

#include <kernels/Conv2D.h>
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
