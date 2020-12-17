/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir2loco.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/SoftmaxOp.h"
#include "mir/ops/TransposeOp.h"

#include <gtest/gtest.h>

class TestTransformer_mir2loco : public ::testing::Test
{
};

TEST_F(TestTransformer_mir2loco, Input_Output_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {5, 6, 7, 8}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(input);
  input->setName("x");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(1));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), pull_node);
  // Shape check
  ASSERT_EQ(pull_node->rank(), 4);
  ASSERT_EQ(pull_node->dim(0), 5);
  ASSERT_EQ(pull_node->dim(1), 6);
  ASSERT_EQ(pull_node->dim(2), 7);
  ASSERT_EQ(pull_node->dim(3), 8);

  ASSERT_TRUE(push_node->indexed());
  ASSERT_EQ(push_node->index(), 0);

  // Check Graph-level properties
  ASSERT_EQ(loco_graph->outputs()->size(), 1);
  ASSERT_NE(loco_graph->outputs()->at(0)->shape(), nullptr);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->rank(), 4);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(0), 5);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(1), 6);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(2), 7);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(3), 8);
}

TEST_F(TestTransformer_mir2loco, Relu_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 9}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *relu = mir_graph.create<mir::ops::ReluOp>(input)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(relu);
  input->setName("x");
  relu->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::ReLU *relu_node = dynamic_cast<loco::ReLU *>(loco_graph->nodes()->at(1));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(2));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(relu_node->input(), pull_node);
  ASSERT_EQ(push_node->from(), relu_node);
  // Shape check
  ASSERT_EQ(pull_node->rank(), 4);
  ASSERT_EQ(pull_node->dim(0), 7);
  ASSERT_EQ(pull_node->dim(1), 7);
  ASSERT_EQ(pull_node->dim(2), 9);
  ASSERT_EQ(pull_node->dim(3), 9);

  ASSERT_TRUE(push_node->indexed());
  ASSERT_EQ(push_node->index(), 0);

  // Check Graph-level properties
  ASSERT_EQ(loco_graph->outputs()->size(), 1);
  ASSERT_NE(loco_graph->outputs()->at(0)->shape(), nullptr);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->rank(), 4);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(0), 7);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(1), 7);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(2), 9);
  ASSERT_EQ(loco_graph->outputs()->at(0)->shape()->dim(3), 9);
}

TEST_F(TestTransformer_mir2loco, Avg_Pool_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 9}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);

  mir::AvgPool2DOpAttributes attributes;
  attributes.window = {2, 3};
  attributes.strides = {4, 5};
  attributes.padding_before = {5, 9};
  attributes.padding_after = {7, 4};
  auto *pool = mir_graph.create<mir::ops::AvgPool2DOp>(input, attributes)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(pool);
  input->setName("x");
  pool->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::FeatureEncode *encode_node =
    dynamic_cast<loco::FeatureEncode *>(loco_graph->nodes()->at(1));
  loco::AvgPool2D *pool_node = dynamic_cast<loco::AvgPool2D *>(loco_graph->nodes()->at(2));
  loco::FeatureDecode *decode_node =
    dynamic_cast<loco::FeatureDecode *>(loco_graph->nodes()->at(3));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(4));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(encode_node, nullptr);
  ASSERT_NE(pool_node, nullptr);
  ASSERT_NE(decode_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  ASSERT_EQ(pool_node->ifm(), encode_node);
  ASSERT_EQ(decode_node->input(), pool_node);
  ASSERT_EQ(push_node->from(), decode_node);
  // Check params
  ASSERT_EQ(pool_node->convention(), loco::AvgPool2D::Convention::Full);
  ASSERT_EQ(pool_node->pad()->top(), 5);
  ASSERT_EQ(pool_node->pad()->left(), 9);
  ASSERT_EQ(pool_node->pad()->bottom(), 7);
  ASSERT_EQ(pool_node->pad()->right(), 4);
  ASSERT_EQ(pool_node->window()->vertical(), 2);
  ASSERT_EQ(pool_node->window()->horizontal(), 3);
  ASSERT_EQ(pool_node->stride()->vertical(), 4);
  ASSERT_EQ(pool_node->stride()->horizontal(), 5);
}

TEST_F(TestTransformer_mir2loco, Max_Pool_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 9}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  mir::MaxPool2DOpAttributes attributes;
  attributes.window = {2, 3};
  attributes.strides = {4, 5};
  attributes.padding_before = {5, 9};
  attributes.padding_after = {7, 4};
  auto *pool = mir_graph.create<mir::ops::MaxPool2DOp>(input, attributes)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(pool);
  input->setName("x");
  pool->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::FeatureEncode *encode_node =
    dynamic_cast<loco::FeatureEncode *>(loco_graph->nodes()->at(1));
  loco::MaxPool2D *pool_node = dynamic_cast<loco::MaxPool2D *>(loco_graph->nodes()->at(2));
  loco::FeatureDecode *decode_node =
    dynamic_cast<loco::FeatureDecode *>(loco_graph->nodes()->at(3));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(4));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(encode_node, nullptr);
  ASSERT_NE(pool_node, nullptr);
  ASSERT_NE(decode_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  ASSERT_EQ(pool_node->ifm(), encode_node);
  ASSERT_EQ(decode_node->input(), pool_node);
  ASSERT_EQ(push_node->from(), decode_node);
  // Check params
  ASSERT_EQ(pool_node->pad()->top(), 5);
  ASSERT_EQ(pool_node->pad()->left(), 9);
  ASSERT_EQ(pool_node->pad()->bottom(), 7);
  ASSERT_EQ(pool_node->pad()->right(), 4);
  ASSERT_EQ(pool_node->window()->vertical(), 2);
  ASSERT_EQ(pool_node->window()->horizontal(), 3);
  ASSERT_EQ(pool_node->stride()->vertical(), 4);
  ASSERT_EQ(pool_node->stride()->horizontal(), 5);
}

TEST_F(TestTransformer_mir2loco, Concat_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {5, 6, 7, 3}};
  auto *input1 = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *input2 = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *input3 = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  std::vector<mir::Operation::Output *> inputs{input1, input2, input3};
  auto *concat = mir_graph.create<mir::ops::ConcatOp>(inputs, 2)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(concat);
  input1->setName("x1");
  input2->setName("x2");
  input3->setName("x3");
  concat->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull1_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::Pull *pull2_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(1));
  loco::Pull *pull3_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(2));
  loco::TensorConcat *concat1_node = dynamic_cast<loco::TensorConcat *>(loco_graph->nodes()->at(3));
  loco::TensorConcat *concat2_node = dynamic_cast<loco::TensorConcat *>(loco_graph->nodes()->at(4));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(5));

  ASSERT_NE(pull1_node, nullptr);
  ASSERT_NE(pull2_node, nullptr);
  ASSERT_NE(pull3_node, nullptr);
  ASSERT_NE(concat1_node, nullptr);
  ASSERT_NE(concat2_node, nullptr);
  ASSERT_NE(push_node, nullptr);

  ASSERT_NE(dynamic_cast<loco::Pull *>(concat1_node->lhs()), nullptr);
  ASSERT_NE(dynamic_cast<loco::Pull *>(concat1_node->rhs()), nullptr);
  ASSERT_EQ(concat2_node->lhs(), concat1_node);
  ASSERT_NE(dynamic_cast<loco::Pull *>(concat2_node->rhs()), nullptr);
  ASSERT_EQ(push_node->from(), concat2_node);
  // Check axis
  ASSERT_EQ(concat1_node->axis(), 2);
  ASSERT_EQ(concat2_node->axis(), 2);
}

TEST_F(TestTransformer_mir2loco, Reshape_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 8, 9, 9}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *reshape = mir_graph.create<mir::ops::ReshapeOp>(input, mir::Shape{7, 8, 81})->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(reshape);
  input->setName("x");
  reshape->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::Reshape<loco::ReshapeType::Fixed> *reshape_node =
    dynamic_cast<loco::Reshape<loco::ReshapeType::Fixed> *>(loco_graph->nodes()->at(1));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(2));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(reshape_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(reshape_node->input(), pull_node);
  ASSERT_EQ(push_node->from(), reshape_node);
  // Check params
  ASSERT_EQ(reshape_node->rank(), 3);
  ASSERT_EQ(reshape_node->dim(0), 7);
  ASSERT_EQ(reshape_node->dim(1), 8);
  ASSERT_EQ(reshape_node->dim(2), 81);
}

TEST_F(TestTransformer_mir2loco, Const_Float_Test)
{
  mir::Graph mir_graph;

  mir::TensorType type{mir::DataType::FLOAT32, {2, 3}};
  const float data[] = {5.9, 6.7, 5.32, 54.11231, 43.2444, 3.409};
  mir::TensorVariant mir_tensor{type, data};
  auto *constant = mir_graph.create<mir::ops::ConstantOp>(mir_tensor)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(constant);
  constant->setName("x");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::ConstGen *const_node = dynamic_cast<loco::ConstGen *>(loco_graph->nodes()->at(0));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(1));

  ASSERT_NE(const_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), const_node);
  // Shape check
  ASSERT_EQ(const_node->rank(), 2);
  ASSERT_EQ(const_node->dim(0), 2);
  ASSERT_EQ(const_node->dim(1), 3);

  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_EQ(const_node->at<loco::DataType::FLOAT32>(i), data[i]);
}

TEST_F(TestTransformer_mir2loco, Add_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input1_type{mir::DataType::FLOAT32, {5, 6, 7, 3}};
  mir::TensorType input2_type{mir::DataType::FLOAT32, {5, 1, 7, 3}};
  auto *input1 = mir_graph.create<mir::ops::InputOp>(input1_type)->getOutput(0);
  auto *input2 = mir_graph.create<mir::ops::InputOp>(input2_type)->getOutput(0);
  auto *add = mir_graph.create<mir::ops::AddOp>(input1, input2)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(add);
  input1->setName("x1");
  input2->setName("x2");
  add->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  // Pull
  auto inputs = loco_graph->inputs();
  ASSERT_EQ(inputs->size(), 2);
  loco::Pull *pull_node0 = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node0, nullptr);
  loco::Pull *pull_node1 = loco::pull_node(loco_graph.get(), 1);
  ASSERT_NE(pull_node1, nullptr);
  // Add
  auto pull_uses = loco::succs(pull_node0);
  ASSERT_EQ(pull_uses.size(), 1);
  loco::EltwiseAdd *add_node = dynamic_cast<loco::EltwiseAdd *>(*pull_uses.begin());
  ASSERT_NE(add_node, nullptr);
  ASSERT_EQ(add_node->lhs(), pull_node0);
  // TensorBroadcast
  loco::TensorBroadcast *broadcast_node = dynamic_cast<loco::TensorBroadcast *>(add_node->rhs());
  ASSERT_NE(broadcast_node, nullptr);
  ASSERT_EQ(broadcast_node->input(), pull_node1);
  // Check params
  ASSERT_TRUE(broadcast_node->mapping()->defined(1));
  ASSERT_EQ(broadcast_node->mapping()->dim(1), 6);
}

TEST_F(TestTransformer_mir2loco, Conv2D_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 1}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);

  mir::TensorType filter_type{mir::DataType::FLOAT32, {2, 3, 1, 1}};
  const float data[] = {5.9, 6.7, 5.32, 54.11231, 43.2444, 3.409};
  mir::TensorVariant filter_tensor{filter_type, data};
  auto *filter = mir_graph.create<mir::ops::ConstantOp>(filter_tensor)->getOutput(0);

  mir::Conv2DOpAttributes attributes;
  attributes.strides = {2, 3};
  attributes.padding_before = {5, 9};
  attributes.padding_after = {7, 4};

  auto *conv = mir_graph.create<mir::ops::Conv2DOp>(input, filter, attributes)->getOutput(0);

  mir_graph.create<mir::ops::OutputOp>(conv);
  input->setName("x");
  conv->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::ConstGen *const_node = dynamic_cast<loco::ConstGen *>(loco_graph->nodes()->at(1));
  loco::FeatureEncode *encode_node =
    dynamic_cast<loco::FeatureEncode *>(loco_graph->nodes()->at(2));
  loco::FilterEncode *filter_node = dynamic_cast<loco::FilterEncode *>(loco_graph->nodes()->at(3));
  loco::Conv2D *conv_node = dynamic_cast<loco::Conv2D *>(loco_graph->nodes()->at(4));
  loco::FeatureDecode *decode_node =
    dynamic_cast<loco::FeatureDecode *>(loco_graph->nodes()->at(5));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(6));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(const_node, nullptr);
  ASSERT_NE(filter_node, nullptr);
  ASSERT_NE(encode_node, nullptr);
  ASSERT_NE(conv_node, nullptr);
  ASSERT_NE(decode_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  ASSERT_EQ(filter_node->input(), const_node);
  ASSERT_EQ(conv_node->ifm(), encode_node);
  ASSERT_EQ(conv_node->ker(), filter_node);
  ASSERT_EQ(decode_node->input(), conv_node);
  ASSERT_EQ(push_node->from(), decode_node);
  // Check params
  ASSERT_EQ(conv_node->pad()->top(), 5);
  ASSERT_EQ(conv_node->pad()->left(), 9);
  ASSERT_EQ(conv_node->pad()->bottom(), 7);
  ASSERT_EQ(conv_node->pad()->right(), 4);
  ASSERT_EQ(conv_node->stride()->vertical(), 2);
  ASSERT_EQ(conv_node->stride()->horizontal(), 3);
}

TEST_F(TestTransformer_mir2loco, Softmax_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 1, 9}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *softmax = mir_graph.create<mir::ops::SoftmaxOp>(input, 2)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(softmax);
  input->setName("x");
  softmax->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  loco::Pull *pull_node = dynamic_cast<loco::Pull *>(loco_graph->nodes()->at(0));
  loco::TensorSoftmax *softmax_node =
    dynamic_cast<loco::TensorSoftmax *>(loco_graph->nodes()->at(1));
  loco::Push *push_node = dynamic_cast<loco::Push *>(loco_graph->nodes()->at(2));

  ASSERT_NE(pull_node, nullptr);
  ASSERT_NE(softmax_node, nullptr);
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(softmax_node->input(), pull_node);
  ASSERT_EQ(push_node->from(), softmax_node);
  // Check axis
  ASSERT_EQ(softmax_node->axis(), 2);
}

TEST_F(TestTransformer_mir2loco, Mul_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input1_type{mir::DataType::FLOAT32, {5, 6, 7, 13}};
  mir::TensorType input2_type{mir::DataType::FLOAT32, {13}};
  auto *input1 = mir_graph.create<mir::ops::InputOp>(input1_type)->getOutput(0);
  auto *input2 = mir_graph.create<mir::ops::InputOp>(input2_type)->getOutput(0);
  auto *add = mir_graph.create<mir::ops::MulOp>(input1, input2)->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(add);
  input1->setName("x1");
  input2->setName("x2");
  add->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  // Pulls
  auto inputs = loco_graph->inputs();
  ASSERT_EQ(inputs->size(), 2);
  loco::Pull *pull_node0 = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node0, nullptr);
  loco::Pull *pull_node1 = loco::pull_node(loco_graph.get(), 1);
  ASSERT_NE(pull_node1, nullptr);
  // Mul
  auto pull0_uses = loco::succs(pull_node0);
  ASSERT_EQ(pull0_uses.size(), 1);
  loco::EltwiseMul *mul_node = dynamic_cast<loco::EltwiseMul *>(*pull0_uses.begin());
  ASSERT_NE(mul_node, nullptr);
  // Broadcast
  loco::TensorBroadcast *broadcast_node = dynamic_cast<loco::TensorBroadcast *>(mul_node->rhs());
  ASSERT_NE(broadcast_node, nullptr);
  ASSERT_EQ(mul_node->lhs(), pull_node0);
  ASSERT_EQ(mul_node->rhs(), broadcast_node);
  loco::FixedReshape *reshape_node = dynamic_cast<loco::FixedReshape *>(broadcast_node->input());
  ASSERT_NE(reshape_node, nullptr);
  ASSERT_EQ(reshape_node->input(), pull_node1);
  ASSERT_EQ(reshape_node->rank(), 4);
  ASSERT_EQ(reshape_node->dim(0), 1);
  ASSERT_EQ(reshape_node->dim(1), 1);
  ASSERT_EQ(reshape_node->dim(2), 1);
  ASSERT_EQ(reshape_node->dim(3), 13);
  // Params checks
  ASSERT_EQ(pull_node0->rank(), 4);
  ASSERT_EQ(pull_node0->dim(0), 5);
  ASSERT_EQ(pull_node0->dim(1), 6);
  ASSERT_EQ(pull_node0->dim(2), 7);
  ASSERT_EQ(pull_node0->dim(3), 13);

  ASSERT_EQ(pull_node1->rank(), 1);
  ASSERT_EQ(pull_node1->dim(0), 13);

  ASSERT_TRUE(broadcast_node->mapping()->defined(0));
  ASSERT_EQ(broadcast_node->mapping()->dim(0), 5);
  ASSERT_TRUE(broadcast_node->mapping()->defined(1));
  ASSERT_EQ(broadcast_node->mapping()->dim(1), 6);
  ASSERT_TRUE(broadcast_node->mapping()->defined(2));
  ASSERT_EQ(broadcast_node->mapping()->dim(2), 7);
}

TEST_F(TestTransformer_mir2loco, DepthwiseConv2D_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 1}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);

  mir::TensorType filter_type{mir::DataType::FLOAT32, {2, 3, 1, 1}};
  const float data[] = {5.9, 6.7, 5.32, 54.11231, 43.2444, 3.409};
  mir::TensorVariant filter_tensor{filter_type, data};
  auto *filter = mir_graph.create<mir::ops::ConstantOp>(filter_tensor)->getOutput(0);

  mir::Conv2DOpAttributes attributes;
  attributes.strides = {2, 3};
  attributes.padding_before = {5, 9};
  attributes.padding_after = {7, 4};

  auto *conv =
    mir_graph.create<mir::ops::DepthwiseConv2DOp>(input, filter, attributes)->getOutput(0);

  mir_graph.create<mir::ops::OutputOp>(conv);
  input->setName("x");
  conv->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  // Pull
  auto inputs = loco_graph->inputs();
  loco::Pull *pull_node = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node, nullptr);
  // FeatureEncode
  auto pull_uses = loco::succs(pull_node);
  ASSERT_EQ(pull_uses.size(), 1);
  loco::FeatureEncode *encode_node = dynamic_cast<loco::FeatureEncode *>(*pull_uses.begin());
  ASSERT_NE(encode_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  // DepthwiseConv2D
  auto encode_uses = loco::succs(encode_node);
  ASSERT_EQ(encode_uses.size(), 1);
  loco::DepthwiseConv2D *dw_conv_node = dynamic_cast<loco::DepthwiseConv2D *>(*encode_uses.begin());
  ASSERT_NE(dw_conv_node, nullptr);
  loco::DepthwiseFilterEncode *filter_node =
    dynamic_cast<loco::DepthwiseFilterEncode *>(dw_conv_node->ker());
  ASSERT_NE(filter_node, nullptr);
  ASSERT_EQ(dw_conv_node->ifm(), encode_node);
  // Check params
  ASSERT_EQ(dw_conv_node->pad()->top(), 5);
  ASSERT_EQ(dw_conv_node->pad()->left(), 9);
  ASSERT_EQ(dw_conv_node->pad()->bottom(), 7);
  ASSERT_EQ(dw_conv_node->pad()->right(), 4);
  ASSERT_EQ(dw_conv_node->stride()->vertical(), 2);
  ASSERT_EQ(dw_conv_node->stride()->horizontal(), 3);
  // ConstGen
  loco::ConstGen *const_node = dynamic_cast<loco::ConstGen *>(filter_node->input());
  ASSERT_NE(const_node, nullptr);
  // FeatureDecode
  auto dw_conv_uses = loco::succs(dw_conv_node);
  ASSERT_EQ(dw_conv_uses.size(), 1);
  loco::FeatureDecode *decode_node = dynamic_cast<loco::FeatureDecode *>(*dw_conv_uses.begin());
  ASSERT_NE(decode_node, nullptr);
  ASSERT_EQ(decode_node->input(), dw_conv_node);
  // Push
  auto decode_uses = loco::succs(decode_node);
  ASSERT_EQ(decode_uses.size(), 1);
  loco::Push *push_node = dynamic_cast<loco::Push *>(*decode_uses.begin());
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), decode_node);
}

TEST_F(TestTransformer_mir2loco, DeConv2D_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {7, 7, 9, 1}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);

  mir::TensorType filter_type{mir::DataType::FLOAT32, {2, 3, 1, 1}};
  const float data[] = {5.9, 6.7, 5.32, 54.11231, 43.2444, 3.409};
  mir::TensorVariant filter_tensor{filter_type, data};
  auto *filter = mir_graph.create<mir::ops::ConstantOp>(filter_tensor)->getOutput(0);

  mir::Deconv2DOpAttributes attributes;
  attributes.strides = {1, 2};
  attributes.padding_before = {3, 4};
  attributes.padding_after = {5, 6};

  auto *conv = mir_graph.create<mir::ops::DeConv2DOp>(input, filter, attributes)->getOutput(0);

  mir_graph.create<mir::ops::OutputOp>(conv);
  input->setName("x");
  conv->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  // Pull
  loco::Pull *pull_node = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node, nullptr);
  // FeatureEncode
  auto pull_uses = loco::succs(pull_node);
  ASSERT_EQ(pull_uses.size(), 1);
  loco::FeatureEncode *encode_node = dynamic_cast<loco::FeatureEncode *>(*pull_uses.begin());
  ASSERT_NE(encode_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  // TransposedConv2D
  auto encode_uses = loco::succs(encode_node);
  ASSERT_EQ(encode_uses.size(), 1);
  loco::TransposedConv2D *tr_conv_node =
    dynamic_cast<loco::TransposedConv2D *>(*encode_uses.begin());
  ASSERT_NE(tr_conv_node, nullptr);
  loco::FilterEncode *filter_node = dynamic_cast<loco::FilterEncode *>(tr_conv_node->ker());
  ASSERT_NE(filter_node, nullptr);
  ASSERT_EQ(tr_conv_node->ifm(), encode_node);
  // Check params
  ASSERT_EQ(tr_conv_node->pad()->top(), 3);
  ASSERT_EQ(tr_conv_node->pad()->left(), 4);
  ASSERT_EQ(tr_conv_node->pad()->bottom(), 5);
  ASSERT_EQ(tr_conv_node->pad()->right(), 6);
  ASSERT_EQ(tr_conv_node->stride()->vertical(), 1);
  ASSERT_EQ(tr_conv_node->stride()->horizontal(), 2);
  // ConstGen
  loco::ConstGen *const_node = dynamic_cast<loco::ConstGen *>(filter_node->input());
  ASSERT_NE(const_node, nullptr);
  // FeatureDecode
  auto tr_conv_uses = loco::succs(tr_conv_node);
  ASSERT_EQ(tr_conv_uses.size(), 1);
  loco::FeatureDecode *decode_node = dynamic_cast<loco::FeatureDecode *>(*tr_conv_uses.begin());
  ASSERT_NE(decode_node, nullptr);
  ASSERT_EQ(decode_node->input(), tr_conv_node);
  // Push
  auto decode_uses = loco::succs(decode_node);
  ASSERT_EQ(decode_uses.size(), 1);
  loco::Push *push_node = dynamic_cast<loco::Push *>(*decode_uses.begin());
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), decode_node);
}

TEST_F(TestTransformer_mir2loco, FullyConnected_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {10, 2}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);

  mir::TensorType weights_type{mir::DataType::FLOAT32, mir::Shape{2, 2}};
  const float data[] = {5.9, 5.32, 54.11231, 3.409};
  mir::TensorVariant weights_tensor{weights_type, data};
  auto *weights = mir_graph.create<mir::ops::ConstantOp>(weights_tensor)->getOutput(0);

  auto *fc = mir_graph.create<mir::ops::FullyConnectedOp>(input, weights)->getOutput(0);

  mir_graph.create<mir::ops::OutputOp>(fc);
  input->setName("x");
  fc->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);

  // Pull
  auto inputs = loco_graph->inputs();
  loco::Pull *pull_node = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node, nullptr);
  // MatrixEncode
  auto pull_uses = loco::succs(pull_node);
  ASSERT_EQ(pull_uses.size(), 1);
  loco::MatrixEncode *encode_node = dynamic_cast<loco::MatrixEncode *>(*pull_uses.begin());
  ASSERT_NE(encode_node, nullptr);
  ASSERT_EQ(encode_node->input(), pull_node);
  // MatMul
  auto encode_uses = loco::succs(encode_node);
  ASSERT_EQ(encode_uses.size(), 1);
  loco::MatMul *fc_node = dynamic_cast<loco::MatMul *>(*encode_uses.begin());
  ASSERT_NE(fc_node, nullptr);
  loco::MatrixEncode *kernel_encode_node = dynamic_cast<loco::MatrixEncode *>(fc_node->rhs());
  ASSERT_NE(kernel_encode_node, nullptr);
  ASSERT_EQ(fc_node->lhs(), encode_node);
  // ConstGen
  loco::ConstGen *const_node = dynamic_cast<loco::ConstGen *>(kernel_encode_node->input());
  ASSERT_NE(const_node, nullptr);
  // MatrixDecode
  auto fc_uses = loco::succs(fc_node);
  ASSERT_EQ(fc_uses.size(), 1);
  loco::MatrixDecode *decode_node = dynamic_cast<loco::MatrixDecode *>(*fc_uses.begin());
  ASSERT_NE(decode_node, nullptr);
  ASSERT_EQ(decode_node->input(), fc_node);
  // Push
  auto decode_uses = loco::succs(decode_node);
  ASSERT_EQ(decode_uses.size(), 1);
  loco::Push *push_node = dynamic_cast<loco::Push *>(*decode_uses.begin());
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), decode_node);
}

TEST_F(TestTransformer_mir2loco, Transpose_Test)
{
  mir::Graph mir_graph;

  mir::TensorType input_type{mir::DataType::FLOAT32, {2, 7, 9, 5}};
  auto *input = mir_graph.create<mir::ops::InputOp>(input_type)->getOutput(0);
  auto *transpose =
    mir_graph.create<mir::ops::TransposeOp>(input, std::vector<std::size_t>{3, 0, 1, 2})
      ->getOutput(0);
  mir_graph.create<mir::ops::OutputOp>(transpose);
  input->setName("x");
  transpose->setName("y");

  mir2loco::Transformer transformer;
  auto loco_graph = transformer.transform(&mir_graph);
  // Pull
  auto inputs = loco_graph->inputs();
  loco::Pull *pull_node = loco::pull_node(loco_graph.get(), 0);
  ASSERT_NE(pull_node, nullptr);
  // Transpose
  auto pull_uses = loco::succs(pull_node);
  ASSERT_EQ(pull_uses.size(), 1);
  loco::TensorTranspose *transpose_node = dynamic_cast<loco::TensorTranspose *>(*pull_uses.begin());
  ASSERT_NE(transpose_node, nullptr);
  ASSERT_EQ(transpose_node->input(), pull_node);
  // Push
  auto transpose_uses = loco::succs(transpose_node);
  ASSERT_EQ(transpose_uses.size(), 1);
  loco::Push *push_node = dynamic_cast<loco::Push *>(*transpose_uses.begin());
  ASSERT_NE(push_node, nullptr);
  ASSERT_EQ(push_node->from(), transpose_node);
  // Axis check
  ASSERT_EQ(transpose_node->perm()->size(), 4);
  ASSERT_EQ(transpose_node->perm()->axis(0), 3);
  ASSERT_EQ(transpose_node->perm()->axis(1), 0);
  ASSERT_EQ(transpose_node->perm()->axis(2), 1);
  ASSERT_EQ(transpose_node->perm()->axis(3), 2);
}
