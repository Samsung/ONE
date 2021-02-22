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

#include "loco/Service/CanonicalShapeInferenceRule.h"
#include "loco/Service/ShapeInference.h"

#include "GraphTestcase.h"

#include <vector>

#include <gtest/gtest.h>

TEST(CanonicalShapeInferenceRuleTest, minimal)
{
  // Create a simple identity network, which takes Tensor<1x2x3x4> as input.
  GraphTestcase<GraphCode::Identity> testcase{1, 2, 3, 4};

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.push_node).domain());
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(1, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1));
  ASSERT_EQ(3, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(2));
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(3));
}

TEST(CanonicalShapeInferenceRuleTest, const_gen)
{
  // Create a sample network
  GraphTestcase<GraphCode::ConstGen> testcase;

  testcase.const_node->dtype(loco::DataType::FLOAT32);
  testcase.const_node->shape({1, 2});

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.push_node).domain());
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(1, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1));
}

TEST(CanonicalShapeInferenceRuleTest, relu)
{
  // Create a sample network
  GraphTestcase<GraphCode::Relu> testcase;

  testcase.pull_node->shape({1, 2, 3, 4});

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.push_node).domain());
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(1, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1));
  ASSERT_EQ(3, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(2));
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(3));
}

TEST(CanonicalShapeInferenceRuleTest, feature_codec)
{
  // Create a sample network
  GraphTestcase<GraphCode::FeatureCodec> testcase;

  testcase.pull_node->shape({1, 2, 3, 4});

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.encode_node));
  ASSERT_EQ(loco::Domain::Feature, loco::shape_get(testcase.encode_node).domain());

  ASSERT_TRUE(loco::shape_known(testcase.decode_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.decode_node).domain());
  ASSERT_EQ(4, loco::shape_get(testcase.decode_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(1, loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(2, loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(1));
  ASSERT_EQ(3, loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(2));
  ASSERT_EQ(4, loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(3));
}

TEST(CanonicalShapeInferenceRuleTest, avgpool2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::AvgPool2D> testcase;

  auto perm = make_NHWC_perm<Domain::Feature>();

  testcase.pull_node->shape({1, 8, 4, 3});

  testcase.encode_node->encoder(std::make_unique<PermutingEncoder<Domain::Feature>>(perm));

  testcase.avgpool2d_node->window()->vertical(2);
  testcase.avgpool2d_node->window()->horizontal(2);

  testcase.avgpool2d_node->stride()->vertical(2);
  testcase.avgpool2d_node->stride()->horizontal(2);

  testcase.decode_node->decoder(std::make_unique<PermutingDecoder<Domain::Feature>>(perm));

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  //
  // NOTE AvgPool2D testcase assumes NHWC layout
  ASSERT_TRUE(loco::shape_known(testcase.avgpool2d_node));
  ASSERT_EQ(loco::Domain::Feature, loco::shape_get(testcase.avgpool2d_node).domain());
  ASSERT_EQ(1, loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().count());
  ASSERT_EQ(3, loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().depth());
  ASSERT_EQ(4, loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().height());
  ASSERT_EQ(2, loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().width());
}

TEST(CanonicalShapeInferenceRuleTest, depthwiseconv2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::DepthwiseConv2D> testcase;

  testcase.pull_node->shape({1, 4, 4, 3});

  testcase.const_node->dtype(loco::DataType::FLOAT32);
  testcase.const_node->shape({2, 2, 3, 2});

  testcase.depthwiseconv2d_node->stride()->vertical(1);
  testcase.depthwiseconv2d_node->stride()->horizontal(1);

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  //
  // NOTE DepthwiseConv2D testcase assumes NHWC layout
  ASSERT_TRUE(loco::shape_known(testcase.depthwiseconv2d_node));
  ASSERT_EQ(loco::Domain::Feature, loco::shape_get(testcase.depthwiseconv2d_node).domain());
  ASSERT_EQ(1, loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().count());
  ASSERT_EQ(6, loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().depth());
  ASSERT_EQ(3, loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().height());
  ASSERT_EQ(3, loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().width());
}

TEST(CanonicalShapeInferenceRuleTest, transposedconv2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::TransposedConv2D> testcase;

  testcase.pull_node->shape({1, 270, 480, 24}); // NHWC

  testcase.const_node->dtype(loco::DataType::FLOAT32);
  testcase.const_node->shape({3, 3, 24, 12}); // HWCN (or HWIO)

  testcase.tr_conv2d_node->stride()->vertical(2);
  testcase.tr_conv2d_node->stride()->horizontal(2);

  testcase.tr_conv2d_node->pad()->top(0);
  testcase.tr_conv2d_node->pad()->bottom(1);
  testcase.tr_conv2d_node->pad()->left(0);
  testcase.tr_conv2d_node->pad()->right(1);

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.tr_conv2d_node));
  ASSERT_EQ(loco::Domain::Feature, loco::shape_get(testcase.tr_conv2d_node).domain());
  ASSERT_EQ(1, loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().count());
  ASSERT_EQ(540, loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().height());
  ASSERT_EQ(960, loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().width());
  ASSERT_EQ(12, loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().depth());
}

TEST(CanonicalShapeInferenceRuleTest, maxpool2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::MaxPool2D> testcase;

  auto perm = make_NHWC_perm<Domain::Feature>();

  testcase.pull_node->shape({1, 8, 4, 3});

  testcase.encode_node->encoder(std::make_unique<PermutingEncoder<Domain::Feature>>(perm));

  testcase.maxpool2d_node->window()->vertical(2);
  testcase.maxpool2d_node->window()->horizontal(2);

  testcase.maxpool2d_node->stride()->vertical(2);
  testcase.maxpool2d_node->stride()->horizontal(2);

  testcase.decode_node->decoder(std::make_unique<PermutingDecoder<Domain::Feature>>(perm));

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  //
  // NOTE MaxPool2D testcase assumes NHWC layout
  ASSERT_TRUE(loco::shape_known(testcase.maxpool2d_node));
  ASSERT_EQ(loco::Domain::Feature, loco::shape_get(testcase.maxpool2d_node).domain());
  ASSERT_EQ(1, loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().count());
  ASSERT_EQ(3, loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().depth());
  ASSERT_EQ(4, loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().height());
  ASSERT_EQ(2, loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().width());
}

TEST(CanonicalShapeInferenceRuleTest, tensor_concat)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::TensorConcat> testcase;

  testcase.lhs_node->shape({1, 2, 3});
  testcase.rhs_node->shape({1, 4, 3});
  testcase.concat_node->axis(1);

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.concat_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.concat_node).domain());
  ASSERT_EQ(3, loco::shape_get(testcase.concat_node).as<TensorShape>().rank());
  ASSERT_EQ(1, loco::shape_get(testcase.concat_node).as<TensorShape>().dim(0));
  ASSERT_EQ(6, loco::shape_get(testcase.concat_node).as<TensorShape>().dim(1));
  ASSERT_EQ(3, loco::shape_get(testcase.concat_node).as<TensorShape>().dim(2));
}

TEST(CanonicalShapeInferenceRuleTest, fixed_reshape)
{
  // Create a sample network
  GraphTestcase<GraphCode::FixedReshape> testcase;

  testcase.pull_node->shape({6, 6});
  testcase.reshape_node->shape({4, 9});

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.push_node).domain());
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(9, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1));
}

TEST(CanonicalShapeInferenceRuleTest, tensor_broadcast)
{
  // Create a sample network
  GraphTestcase<GraphCode::TensorBroadcast> testcase{1, 2};

  testcase.broadcast_node->mapping()->dim(0) = 4;

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(testcase.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(testcase.push_node).domain());
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(4, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(2, loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1));
}

TEST(CanonicalShapeInferenceRuleTest, tensor_transpose)
{
  // Create a sample network
  GraphTestcase<GraphCode::TensorTranspose> tc;

  tc.pull_node->shape({10, 20, 30, 40});

  tc.transpose_node->perm()->size(4);
  tc.transpose_node->perm()->axis(0) = 2;
  tc.transpose_node->perm()->axis(1) = 3;
  tc.transpose_node->perm()->axis(2) = 0;
  tc.transpose_node->perm()->axis(3) = 1;

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(tc.graph());

  // Verify!
  ASSERT_TRUE(loco::shape_known(tc.push_node));
  ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(tc.push_node).domain());
  ASSERT_EQ(4, loco::shape_get(tc.push_node).as<loco::TensorShape>().rank());
  ASSERT_EQ(30, loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(0));
  ASSERT_EQ(40, loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(1));
  ASSERT_EQ(10, loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(2));
  ASSERT_EQ(20, loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(3));
}

namespace
{

struct MockContext final : public loco::ShapeInferenceRule::Context
{
  bool known(const loco::Node *node) const final { return _content.find(node) != _content.end(); }
  loco::NodeShape get(const loco::Node *node) const final { return _content.at(node); }

  std::map<const loco::Node *, loco::NodeShape> _content;
};

struct MockSink final : public loco::ShapeInferenceRule::Sink
{
  void okay(const loco::NodeShape &res) final { shape = res; }
  void fail(void) final { return; }

  loco::NodeShape shape;
};

} // namespace

TEST(CanonicalShapeInferenceRuleTest, infer_v2)
{
  auto g = loco::make_graph();

  // Create an incomplete graph
  auto relu_1 = g->nodes()->create<loco::ReLU>();
  auto relu_2 = g->nodes()->create<loco::ReLU>();

  relu_2->input(relu_1);

  // Set up Context
  MockContext ctx;

  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);
  tensor_shape.dim(0) = 4;
  tensor_shape.dim(1) = 5;

  ctx._content[relu_1] = tensor_shape;

  // Create a Sink
  MockSink sink;

  loco::CanonicalShapeInferenceRule rule;

  rule.infer(&ctx, relu_2, &sink);

  ASSERT_EQ(loco::Domain::Tensor, sink.shape.domain());
  ASSERT_EQ(2, sink.shape.as<loco::TensorShape>().rank());
  ASSERT_EQ(4, sink.shape.as<loco::TensorShape>().dim(0));
  ASSERT_EQ(5, sink.shape.as<loco::TensorShape>().dim(1));
}
