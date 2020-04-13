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
  ASSERT_EQ(loco::shape_get(testcase.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank(), 4);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0), 1);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1), 2);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(2), 3);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(3), 4);
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
  ASSERT_EQ(loco::shape_get(testcase.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0), 1);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1), 2);
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
  ASSERT_EQ(loco::shape_get(testcase.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank(), 4);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0), 1);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1), 2);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(2), 3);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(3), 4);
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
  ASSERT_EQ(loco::shape_get(testcase.encode_node).domain(), loco::Domain::Feature);

  ASSERT_TRUE(loco::shape_known(testcase.decode_node));
  ASSERT_EQ(loco::shape_get(testcase.decode_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.decode_node).as<loco::TensorShape>().rank(), 4);
  ASSERT_EQ(loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(0), 1);
  ASSERT_EQ(loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(1), 2);
  ASSERT_EQ(loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(2), 3);
  ASSERT_EQ(loco::shape_get(testcase.decode_node).as<loco::TensorShape>().dim(3), 4);
}

TEST(CanonicalShapeInferenceRuleTest, avgpool2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::AvgPool2D> testcase;

  auto perm = make_NHWC_perm<Domain::Feature>();

  testcase.pull_node->shape({1, 8, 4, 3});

  testcase.encode_node->encoder(stdex::make_unique<PermutingEncoder<Domain::Feature>>(perm));

  testcase.avgpool2d_node->window()->vertical(2);
  testcase.avgpool2d_node->window()->horizontal(2);

  testcase.avgpool2d_node->stride()->vertical(2);
  testcase.avgpool2d_node->stride()->horizontal(2);

  testcase.decode_node->decoder(stdex::make_unique<PermutingDecoder<Domain::Feature>>(perm));

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  //
  // NOTE AvgPool2D testcase assumes NHWC layout
  ASSERT_TRUE(loco::shape_known(testcase.avgpool2d_node));
  ASSERT_EQ(loco::shape_get(testcase.avgpool2d_node).domain(), loco::Domain::Feature);
  ASSERT_EQ(loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().count(), 1);
  ASSERT_EQ(loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().depth(), 3);
  ASSERT_EQ(loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().height(), 4);
  ASSERT_EQ(loco::shape_get(testcase.avgpool2d_node).as<FeatureShape>().width(), 2);
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
  ASSERT_EQ(loco::shape_get(testcase.depthwiseconv2d_node).domain(), loco::Domain::Feature);
  ASSERT_EQ(loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().count(), 1);
  ASSERT_EQ(loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().depth(), 6);
  ASSERT_EQ(loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().height(), 3);
  ASSERT_EQ(loco::shape_get(testcase.depthwiseconv2d_node).as<FeatureShape>().width(), 3);
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
  ASSERT_EQ(loco::shape_get(testcase.tr_conv2d_node).domain(), loco::Domain::Feature);
  ASSERT_EQ(loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().count(), 1);
  ASSERT_EQ(loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().height(), 540);
  ASSERT_EQ(loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().width(), 960);
  ASSERT_EQ(loco::shape_get(testcase.tr_conv2d_node).as<FeatureShape>().depth(), 12);
}

TEST(CanonicalShapeInferenceRuleTest, maxpool2d)
{
  using namespace loco;

  // Create a sample network
  GraphTestcase<GraphCode::MaxPool2D> testcase;

  auto perm = make_NHWC_perm<Domain::Feature>();

  testcase.pull_node->shape({1, 8, 4, 3});

  testcase.encode_node->encoder(stdex::make_unique<PermutingEncoder<Domain::Feature>>(perm));

  testcase.maxpool2d_node->window()->vertical(2);
  testcase.maxpool2d_node->window()->horizontal(2);

  testcase.maxpool2d_node->stride()->vertical(2);
  testcase.maxpool2d_node->stride()->horizontal(2);

  testcase.decode_node->decoder(stdex::make_unique<PermutingDecoder<Domain::Feature>>(perm));

  // Run Inference
  loco::CanonicalShapeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  //
  // NOTE MaxPool2D testcase assumes NHWC layout
  ASSERT_TRUE(loco::shape_known(testcase.maxpool2d_node));
  ASSERT_EQ(loco::shape_get(testcase.maxpool2d_node).domain(), loco::Domain::Feature);
  ASSERT_EQ(loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().count(), 1);
  ASSERT_EQ(loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().depth(), 3);
  ASSERT_EQ(loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().height(), 4);
  ASSERT_EQ(loco::shape_get(testcase.maxpool2d_node).as<FeatureShape>().width(), 2);
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
  ASSERT_EQ(loco::shape_get(testcase.concat_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.concat_node).as<TensorShape>().rank(), 3);
  ASSERT_EQ(loco::shape_get(testcase.concat_node).as<TensorShape>().dim(0), 1);
  ASSERT_EQ(loco::shape_get(testcase.concat_node).as<TensorShape>().dim(1), 6);
  ASSERT_EQ(loco::shape_get(testcase.concat_node).as<TensorShape>().dim(2), 3);
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
  ASSERT_EQ(loco::shape_get(testcase.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0), 4);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1), 9);
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
  ASSERT_EQ(loco::shape_get(testcase.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(0), 4);
  ASSERT_EQ(loco::shape_get(testcase.push_node).as<loco::TensorShape>().dim(1), 2);
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
  ASSERT_EQ(loco::shape_get(tc.push_node).domain(), loco::Domain::Tensor);
  ASSERT_EQ(loco::shape_get(tc.push_node).as<loco::TensorShape>().rank(), 4);
  ASSERT_EQ(loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(0), 30);
  ASSERT_EQ(loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(1), 40);
  ASSERT_EQ(loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(2), 10);
  ASSERT_EQ(loco::shape_get(tc.push_node).as<loco::TensorShape>().dim(3), 20);
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

  ASSERT_EQ(sink.shape.domain(), loco::Domain::Tensor);
  ASSERT_EQ(sink.shape.as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(sink.shape.as<loco::TensorShape>().dim(0), 4);
  ASSERT_EQ(sink.shape.as<loco::TensorShape>().dim(1), 5);
}
