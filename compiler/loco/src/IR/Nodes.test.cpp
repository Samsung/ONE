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

#include "loco/IR/Nodes.h"
#include "loco/IR/CanonicalDialect.h"

#include <gtest/gtest.h>

TEST(PushTest, constructor)
{
  loco::Push push_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), push_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::Push, push_node.opcode());

  ASSERT_FALSE(push_node.indexed());
}

TEST(PushTest, shape)
{
  const std::vector<uint32_t> dims{1, 8, 16, 3};

  loco::Pull push_node;

  push_node.shape({dims[0], dims[1], dims[2], dims[3]});

  ASSERT_EQ(dims.size(), push_node.rank());
  ASSERT_EQ(dims[0], push_node.dim(0));
  ASSERT_EQ(dims[1], push_node.dim(1));
  ASSERT_EQ(dims[2], push_node.dim(2));
  ASSERT_EQ(dims[3], push_node.dim(3));
}

TEST(PullTest, constructor)
{
  loco::Pull pull_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), pull_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::Pull, pull_node.opcode());

  ASSERT_FALSE(pull_node.indexed());

  ASSERT_EQ(loco::DataType::Unknown, pull_node.dtype());
  ASSERT_EQ(0, pull_node.rank());
}

TEST(PullTest, shape)
{
  const std::vector<uint32_t> dims{1, 8, 16, 3};

  loco::Pull pull_node;

  pull_node.shape({dims[0], dims[1], dims[2], dims[3]});

  ASSERT_EQ(dims.size(), pull_node.rank());
  ASSERT_EQ(dims[0], pull_node.dim(0));
  ASSERT_EQ(dims[1], pull_node.dim(1));
  ASSERT_EQ(dims[2], pull_node.dim(2));
  ASSERT_EQ(dims[3], pull_node.dim(3));
}

TEST(ForwardTest, constructor)
{
  loco::Forward forward_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), forward_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::Forward, forward_node.opcode());

  ASSERT_EQ(nullptr, forward_node.input());
}

TEST(ReLUTest, constructor)
{
  loco::ReLU relu_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), relu_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::ReLU, relu_node.opcode());

  ASSERT_EQ(nullptr, relu_node.input());
}

TEST(ReLU6Test, constructor)
{
  loco::ReLU6 relu6_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), relu6_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::ReLU6, relu6_node.opcode());

  ASSERT_EQ(nullptr, relu6_node.input());
}

TEST(ConstGenTest, constructor)
{
  loco::ConstGen constgen_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), constgen_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::ConstGen, constgen_node.opcode());

  ASSERT_EQ(loco::DataType::Unknown, constgen_node.dtype());
  ASSERT_EQ(0, constgen_node.rank());

  constgen_node.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(loco::DataType::FLOAT32, constgen_node.dtype());

  constgen_node.rank(2);
  ASSERT_EQ(2, constgen_node.rank());

  constgen_node.dim(0) = 2;
  constgen_node.dim(1) = 3;

  ASSERT_TRUE(constgen_node.dim(0).known());
  ASSERT_TRUE(constgen_node.dim(1).known());

  ASSERT_EQ(2, constgen_node.dim(0));
  ASSERT_EQ(3, constgen_node.dim(1));

  constgen_node.size<loco::DataType::FLOAT32>(6);

  ASSERT_EQ(6, constgen_node.size<loco::DataType::FLOAT32>());

  constgen_node.at<loco::DataType::FLOAT32>(0) = 0.0f; // Set 0,0
  constgen_node.at<loco::DataType::FLOAT32>(1) = 1.0f; // Set 0,1
  constgen_node.at<loco::DataType::FLOAT32>(2) = 2.0f; // Set 0,2
  constgen_node.at<loco::DataType::FLOAT32>(3) = 3.0f; // Set 1,0
  constgen_node.at<loco::DataType::FLOAT32>(4) = 4.0f; // Set 1,1
  constgen_node.at<loco::DataType::FLOAT32>(5) = 5.0f; // Set 1,2

  ASSERT_EQ(0.0f, constgen_node.at<loco::DataType::FLOAT32>(0));
  ASSERT_EQ(1.0f, constgen_node.at<loco::DataType::FLOAT32>(1));
  ASSERT_EQ(2.0f, constgen_node.at<loco::DataType::FLOAT32>(2));
  ASSERT_EQ(3.0f, constgen_node.at<loco::DataType::FLOAT32>(3));
  ASSERT_EQ(4.0f, constgen_node.at<loco::DataType::FLOAT32>(4));
  ASSERT_EQ(5.0f, constgen_node.at<loco::DataType::FLOAT32>(5));
}

TEST(ConstGenTest, constructor_s32)
{
  loco::ConstGen constgen_node;

  ASSERT_EQ(loco::DataType::Unknown, constgen_node.dtype());
  ASSERT_EQ(0, constgen_node.rank());

  constgen_node.dtype(loco::DataType::S32);
  ASSERT_EQ(loco::DataType::S32, constgen_node.dtype());

  constgen_node.rank(2);
  ASSERT_EQ(2, constgen_node.rank());

  constgen_node.dim(0) = 2;
  constgen_node.dim(1) = 3;

  ASSERT_TRUE(constgen_node.dim(0).known());
  ASSERT_TRUE(constgen_node.dim(1).known());

  ASSERT_EQ(2, constgen_node.dim(0));
  ASSERT_EQ(3, constgen_node.dim(1));

  constgen_node.size<loco::DataType::S32>(6);

  ASSERT_EQ(6, constgen_node.size<loco::DataType::S32>());

  constgen_node.at<loco::DataType::S32>(0) = 0;  // Set 0,0
  constgen_node.at<loco::DataType::S32>(1) = 1;  // Set 0,1
  constgen_node.at<loco::DataType::S32>(2) = 2;  // Set 0,2
  constgen_node.at<loco::DataType::S32>(3) = -3; // Set 1,0
  constgen_node.at<loco::DataType::S32>(4) = -4; // Set 1,1
  constgen_node.at<loco::DataType::S32>(5) = -5; // Set 1,2

  ASSERT_EQ(0, constgen_node.at<loco::DataType::S32>(0));
  ASSERT_EQ(1, constgen_node.at<loco::DataType::S32>(1));
  ASSERT_EQ(2, constgen_node.at<loco::DataType::S32>(2));
  ASSERT_EQ(-3, constgen_node.at<loco::DataType::S32>(3));
  ASSERT_EQ(-4, constgen_node.at<loco::DataType::S32>(4));
  ASSERT_EQ(-5, constgen_node.at<loco::DataType::S32>(5));
}

TEST(MaxPool2DTest, constructor)
{
  loco::MaxPool2D maxpool_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), maxpool_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::MaxPool2D, maxpool_node.opcode());

  ASSERT_EQ(nullptr, maxpool_node.ifm());

  ASSERT_EQ(0, maxpool_node.pad()->top());
  ASSERT_EQ(0, maxpool_node.pad()->bottom());
  ASSERT_EQ(0, maxpool_node.pad()->left());
  ASSERT_EQ(0, maxpool_node.pad()->right());

  ASSERT_EQ(1, maxpool_node.window()->vertical());
  ASSERT_EQ(1, maxpool_node.window()->horizontal());

  ASSERT_EQ(1, maxpool_node.stride()->vertical());
  ASSERT_EQ(1, maxpool_node.stride()->horizontal());
}

TEST(MaxPool2DTest, pad)
{
  const uint32_t t = 1;
  const uint32_t b = 2;
  const uint32_t l = 3;
  const uint32_t r = 4;

  loco::MaxPool2D maxpool_node;

  maxpool_node.pad()->top(t);
  ASSERT_EQ(t, maxpool_node.pad()->top());

  maxpool_node.pad()->bottom(b);
  ASSERT_EQ(b, maxpool_node.pad()->bottom());

  maxpool_node.pad()->left(l);
  ASSERT_EQ(l, maxpool_node.pad()->left());

  maxpool_node.pad()->right(r);
  ASSERT_EQ(r, maxpool_node.pad()->right());
}

TEST(AvgPool2DTest, constructor)
{
  loco::AvgPool2D avgpool_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), avgpool_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::AvgPool2D, avgpool_node.opcode());

  ASSERT_EQ(nullptr, avgpool_node.ifm());

  ASSERT_EQ(loco::AvgPool2D::Convention::Unknown, avgpool_node.convention());

  ASSERT_EQ(0, avgpool_node.pad()->top());
  ASSERT_EQ(0, avgpool_node.pad()->bottom());
  ASSERT_EQ(0, avgpool_node.pad()->left());
  ASSERT_EQ(0, avgpool_node.pad()->right());

  ASSERT_EQ(1, avgpool_node.window()->vertical());
  ASSERT_EQ(1, avgpool_node.window()->horizontal());

  ASSERT_EQ(1, avgpool_node.stride()->vertical());
  ASSERT_EQ(1, avgpool_node.stride()->horizontal());
}

TEST(FeatureEncodeTest, constructor)
{
  loco::FeatureEncode feature_encode;

  ASSERT_EQ(loco::CanonicalDialect::get(), feature_encode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FeatureEncode, feature_encode.opcode());

  ASSERT_EQ(nullptr, feature_encode.input());
  ASSERT_EQ(nullptr, feature_encode.encoder());
}

TEST(FeatureDecodeTest, constructor)
{
  loco::FeatureDecode feature_decode;

  ASSERT_EQ(loco::CanonicalDialect::get(), feature_decode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FeatureDecode, feature_decode.opcode());

  ASSERT_EQ(nullptr, feature_decode.input());
  ASSERT_EQ(nullptr, feature_decode.decoder());
}

TEST(Reshape_Fixed_Test, constructor)
{
  loco::Reshape<loco::ReshapeType::Fixed> reshape;

  ASSERT_EQ(loco::CanonicalDialect::get(), reshape.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FixedReshape, reshape.opcode());

  ASSERT_EQ(0, reshape.rank());
}

TEST(Reshape_Fixed_Test, shape)
{
  loco::Reshape<loco::ReshapeType::Fixed> reshape;
  reshape.shape({2, 3});

  ASSERT_EQ(2, reshape.rank());
  ASSERT_EQ(2, reshape.dim(0));
  ASSERT_EQ(3, reshape.dim(1));
}

TEST(FilterEncodeTest, constructor)
{
  loco::FilterEncode filter_encode;

  ASSERT_EQ(loco::CanonicalDialect::get(), filter_encode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FilterEncode, filter_encode.opcode());

  ASSERT_EQ(nullptr, filter_encode.input());
  ASSERT_EQ(nullptr, filter_encode.encoder());
}

TEST(FilterDecodeTest, constructor)
{
  loco::FilterDecode filter_decode;

  ASSERT_EQ(loco::CanonicalDialect::get(), filter_decode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FilterDecode, filter_decode.opcode());

  ASSERT_EQ(nullptr, filter_decode.input());
  ASSERT_EQ(nullptr, filter_decode.decoder());
}

TEST(DepthwiseFilterEncodeTest, constructor)
{
  loco::DepthwiseFilterEncode dw_filter_encode;

  ASSERT_EQ(loco::CanonicalDialect::get(), dw_filter_encode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::DepthwiseFilterEncode, dw_filter_encode.opcode());

  ASSERT_EQ(nullptr, dw_filter_encode.input());
  ASSERT_EQ(nullptr, dw_filter_encode.encoder());
}

TEST(DepthwiseFilterDecodeTest, constructor)
{
  loco::DepthwiseFilterDecode dw_filter_decode;

  ASSERT_EQ(loco::CanonicalDialect::get(), dw_filter_decode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::DepthwiseFilterDecode, dw_filter_decode.opcode());

  ASSERT_EQ(nullptr, dw_filter_decode.input());
  ASSERT_EQ(nullptr, dw_filter_decode.decoder());
}

TEST(TensorConcatTest, constructor)
{
  loco::TensorConcat tensor_concat;

  ASSERT_EQ(loco::CanonicalDialect::get(), tensor_concat.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::TensorConcat, tensor_concat.opcode());

  ASSERT_EQ(nullptr, tensor_concat.lhs());
  ASSERT_EQ(nullptr, tensor_concat.rhs());
  ASSERT_EQ(0, tensor_concat.axis());

  tensor_concat.axis(3);
  ASSERT_EQ(3, tensor_concat.axis());
}

TEST(Conv2DTest, constructor)
{
  loco::Conv2D conv2d;

  ASSERT_EQ(loco::CanonicalDialect::get(), conv2d.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::Conv2D, conv2d.opcode());

  ASSERT_EQ(nullptr, conv2d.ifm());
  ASSERT_EQ(nullptr, conv2d.ker());

  ASSERT_NE(conv2d.pad(), nullptr);
  ASSERT_EQ(0, conv2d.pad()->top());
  ASSERT_EQ(0, conv2d.pad()->bottom());
  ASSERT_EQ(0, conv2d.pad()->left());
  ASSERT_EQ(0, conv2d.pad()->right());

  ASSERT_NE(conv2d.stride(), nullptr);
  ASSERT_EQ(1, conv2d.stride()->vertical());
  ASSERT_EQ(1, conv2d.stride()->horizontal());
}

TEST(DepthwiseConv2DTest, constructor)
{
  loco::DepthwiseConv2D dw_conv2d;

  ASSERT_EQ(loco::CanonicalDialect::get(), dw_conv2d.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::DepthwiseConv2D, dw_conv2d.opcode());

  ASSERT_EQ(nullptr, dw_conv2d.ifm());
  ASSERT_EQ(nullptr, dw_conv2d.ker());

  ASSERT_NE(dw_conv2d.pad(), nullptr);
  ASSERT_EQ(0, dw_conv2d.pad()->top());
  ASSERT_EQ(0, dw_conv2d.pad()->bottom());
  ASSERT_EQ(0, dw_conv2d.pad()->left());
  ASSERT_EQ(0, dw_conv2d.pad()->right());

  ASSERT_NE(dw_conv2d.stride(), nullptr);
  ASSERT_EQ(1, dw_conv2d.stride()->vertical());
  ASSERT_EQ(1, dw_conv2d.stride()->horizontal());
}

TEST(TransposedConv2DTest, constructor)
{
  loco::TransposedConv2D tr_conv2d;

  ASSERT_EQ(loco::CanonicalDialect::get(), tr_conv2d.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::TransposedConv2D, tr_conv2d.opcode());

  ASSERT_EQ(nullptr, tr_conv2d.ifm());
  ASSERT_EQ(nullptr, tr_conv2d.ker());

  ASSERT_NE(tr_conv2d.pad(), nullptr);
  ASSERT_EQ(0, tr_conv2d.pad()->top());
  ASSERT_EQ(0, tr_conv2d.pad()->bottom());
  ASSERT_EQ(0, tr_conv2d.pad()->left());
  ASSERT_EQ(0, tr_conv2d.pad()->right());

  ASSERT_NE(tr_conv2d.stride(), nullptr);
  ASSERT_EQ(1, tr_conv2d.stride()->vertical());
  ASSERT_EQ(1, tr_conv2d.stride()->horizontal());
}

TEST(BiasEncodeTest, constructor)
{
  loco::BiasEncode bias_encode;

  ASSERT_EQ(loco::CanonicalDialect::get(), bias_encode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::BiasEncode, bias_encode.opcode());

  ASSERT_EQ(nullptr, bias_encode.input());
}

TEST(TensorBiasAddTest, constructor)
{
  loco::BiasAdd<loco::Domain::Tensor> bias_add;

  ASSERT_EQ(loco::CanonicalDialect::get(), bias_add.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::TensorBiasAdd, bias_add.opcode());

  ASSERT_EQ(nullptr, bias_add.value());
  ASSERT_EQ(nullptr, bias_add.bias());
  ASSERT_EQ(0, bias_add.axis());
}

TEST(TensorBiasAddTest, alias)
{
  loco::TensorBiasAdd bias_add;

  SUCCEED();
}

TEST(FeatureBiasAddTest, constructor)
{
  loco::BiasAdd<loco::Domain::Feature> bias_add;

  ASSERT_EQ(loco::CanonicalDialect::get(), bias_add.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::FeatureBiasAdd, bias_add.opcode());

  ASSERT_EQ(nullptr, bias_add.value());
  ASSERT_EQ(nullptr, bias_add.bias());
}

TEST(FeatureBiasAddTest, alias)
{
  loco::FeatureBiasAdd bias_add;

  SUCCEED();
}

TEST(EltwiseAddTest, constructor)
{
  loco::EltwiseAdd eltwise_add;

  SUCCEED();
}

TEST(EltwiseMaxTest, constructor)
{
  loco::EltwiseMax eltwise_max;

  SUCCEED();
}

TEST(EltwiseMulTest, constructor)
{
  loco::EltwiseMul eltwise_mul;

  SUCCEED();
}

TEST(EltwiseSubTest, constructor)
{
  loco::EltwiseSub eltwise_sub;

  SUCCEED();
}

TEST(EltwiseDivTest, constructor)
{
  loco::EltwiseDiv eltwise_div;

  SUCCEED();
}

TEST(EltwiseSqrtTest, constructor)
{
  loco::EltwiseSqrt sqrt_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), sqrt_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::EltwiseSqrt, sqrt_node.opcode());

  ASSERT_EQ(nullptr, sqrt_node.input());
}

TEST(TensorBroadcastTest, constructor)
{
  loco::TensorBroadcast tensor_broadcast_node;

  ASSERT_EQ(loco::CanonicalDialect::get(), tensor_broadcast_node.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::TensorBroadcast, tensor_broadcast_node.opcode());

  ASSERT_EQ(nullptr, tensor_broadcast_node.input());
}

TEST(TensorBroadcastTest, mapping)
{
  loco::TensorBroadcast tensor_broadcast_node;

  ASSERT_FALSE(tensor_broadcast_node.mapping()->defined(0));

  tensor_broadcast_node.mapping()->dim(0) = 3;

  ASSERT_TRUE(tensor_broadcast_node.mapping()->defined(0));
  ASSERT_EQ(3, tensor_broadcast_node.mapping()->dim(0));
}

TEST(MatrixEncodeTest, constructor)
{
  loco::MatrixEncode matrix_encode;

  ASSERT_EQ(loco::CanonicalDialect::get(), matrix_encode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::MatrixEncode, matrix_encode.opcode());

  ASSERT_EQ(nullptr, matrix_encode.input());
}

TEST(MatrixDecodeTest, constructor)
{
  loco::MatrixDecode matrix_decode;

  ASSERT_EQ(loco::CanonicalDialect::get(), matrix_decode.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::MatrixDecode, matrix_decode.opcode());

  ASSERT_EQ(nullptr, matrix_decode.input());
}

TEST(MatMulTest, constructor)
{
  loco::MatMul mat_mul;

  ASSERT_EQ(loco::CanonicalDialect::get(), mat_mul.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::MatMul, mat_mul.opcode());

  ASSERT_EQ(nullptr, mat_mul.lhs());
  ASSERT_EQ(nullptr, mat_mul.rhs());
}

TEST(TransposeTest, constructor)
{
  loco::TensorTranspose transpose;

  ASSERT_EQ(loco::CanonicalDialect::get(), transpose.dialect());
  ASSERT_EQ(loco::CanonicalOpcode::TensorTranspose, transpose.opcode());

  ASSERT_EQ(nullptr, transpose.input());
  ASSERT_EQ(0, transpose.perm()->size());
}

TEST(TransposeTest, perm)
{
  loco::TensorTranspose transpose;

  transpose.perm()->size(3);
  transpose.perm()->axis(0) = 1;
  transpose.perm()->axis(1) = 2;
  transpose.perm()->axis(2) = 0;

  ASSERT_EQ(1, transpose.perm()->axis(0));
  ASSERT_EQ(2, transpose.perm()->axis(1));
  ASSERT_EQ(0, transpose.perm()->axis(2));
}
