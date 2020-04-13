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

  ASSERT_EQ(push_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(push_node.opcode(), loco::CanonicalOpcode::Push);

  ASSERT_FALSE(push_node.indexed());
}

TEST(PushTest, shape)
{
  const std::vector<uint32_t> dims{1, 8, 16, 3};

  loco::Pull push_node;

  push_node.shape({dims[0], dims[1], dims[2], dims[3]});

  ASSERT_EQ(push_node.rank(), dims.size());
  ASSERT_EQ(push_node.dim(0), dims[0]);
  ASSERT_EQ(push_node.dim(1), dims[1]);
  ASSERT_EQ(push_node.dim(2), dims[2]);
  ASSERT_EQ(push_node.dim(3), dims[3]);
}

TEST(PullTest, constructor)
{
  loco::Pull pull_node;

  ASSERT_EQ(pull_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(pull_node.opcode(), loco::CanonicalOpcode::Pull);

  ASSERT_FALSE(pull_node.indexed());

  ASSERT_EQ(pull_node.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(pull_node.rank(), 0);
}

TEST(PullTest, shape)
{
  const std::vector<uint32_t> dims{1, 8, 16, 3};

  loco::Pull pull_node;

  pull_node.shape({dims[0], dims[1], dims[2], dims[3]});

  ASSERT_EQ(pull_node.rank(), dims.size());
  ASSERT_EQ(pull_node.dim(0), dims[0]);
  ASSERT_EQ(pull_node.dim(1), dims[1]);
  ASSERT_EQ(pull_node.dim(2), dims[2]);
  ASSERT_EQ(pull_node.dim(3), dims[3]);
}

TEST(ForwardTest, constructor)
{
  loco::Forward forward_node;

  ASSERT_EQ(forward_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(forward_node.opcode(), loco::CanonicalOpcode::Forward);

  ASSERT_EQ(forward_node.input(), nullptr);
}

TEST(ReLUTest, constructor)
{
  loco::ReLU relu_node;

  ASSERT_EQ(relu_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(relu_node.opcode(), loco::CanonicalOpcode::ReLU);

  ASSERT_EQ(relu_node.input(), nullptr);
}

TEST(ReLU6Test, constructor)
{
  loco::ReLU6 relu6_node;

  ASSERT_EQ(relu6_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(relu6_node.opcode(), loco::CanonicalOpcode::ReLU6);

  ASSERT_EQ(relu6_node.input(), nullptr);
}

TEST(ConstGenTest, constructor)
{
  loco::ConstGen constgen_node;

  ASSERT_EQ(constgen_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(constgen_node.opcode(), loco::CanonicalOpcode::ConstGen);

  ASSERT_EQ(constgen_node.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(constgen_node.rank(), 0);

  constgen_node.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(constgen_node.dtype(), loco::DataType::FLOAT32);

  constgen_node.rank(2);
  ASSERT_EQ(constgen_node.rank(), 2);

  constgen_node.dim(0) = 2;
  constgen_node.dim(1) = 3;

  ASSERT_TRUE(constgen_node.dim(0).known());
  ASSERT_TRUE(constgen_node.dim(1).known());

  ASSERT_EQ(constgen_node.dim(0), 2);
  ASSERT_EQ(constgen_node.dim(1), 3);

  constgen_node.size<loco::DataType::FLOAT32>(6);

  ASSERT_EQ(constgen_node.size<loco::DataType::FLOAT32>(), 6);

  constgen_node.at<loco::DataType::FLOAT32>(0) = 0.0f; // Set 0,0
  constgen_node.at<loco::DataType::FLOAT32>(1) = 1.0f; // Set 0,1
  constgen_node.at<loco::DataType::FLOAT32>(2) = 2.0f; // Set 0,2
  constgen_node.at<loco::DataType::FLOAT32>(3) = 3.0f; // Set 1,0
  constgen_node.at<loco::DataType::FLOAT32>(4) = 4.0f; // Set 1,1
  constgen_node.at<loco::DataType::FLOAT32>(5) = 5.0f; // Set 1,2

  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(0), 0.0f);
  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(1), 1.0f);
  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(2), 2.0f);
  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(3), 3.0f);
  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(4), 4.0f);
  ASSERT_EQ(constgen_node.at<loco::DataType::FLOAT32>(5), 5.0f);
}

TEST(ConstGenTest, constructor_s32)
{
  loco::ConstGen constgen_node;

  ASSERT_EQ(constgen_node.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(constgen_node.rank(), 0);

  constgen_node.dtype(loco::DataType::S32);
  ASSERT_EQ(constgen_node.dtype(), loco::DataType::S32);

  constgen_node.rank(2);
  ASSERT_EQ(constgen_node.rank(), 2);

  constgen_node.dim(0) = 2;
  constgen_node.dim(1) = 3;

  ASSERT_TRUE(constgen_node.dim(0).known());
  ASSERT_TRUE(constgen_node.dim(1).known());

  ASSERT_EQ(constgen_node.dim(0), 2);
  ASSERT_EQ(constgen_node.dim(1), 3);

  constgen_node.size<loco::DataType::S32>(6);

  ASSERT_EQ(constgen_node.size<loco::DataType::S32>(), 6);

  constgen_node.at<loco::DataType::S32>(0) = 0;  // Set 0,0
  constgen_node.at<loco::DataType::S32>(1) = 1;  // Set 0,1
  constgen_node.at<loco::DataType::S32>(2) = 2;  // Set 0,2
  constgen_node.at<loco::DataType::S32>(3) = -3; // Set 1,0
  constgen_node.at<loco::DataType::S32>(4) = -4; // Set 1,1
  constgen_node.at<loco::DataType::S32>(5) = -5; // Set 1,2

  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(0), 0);
  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(1), 1);
  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(2), 2);
  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(3), -3);
  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(4), -4);
  ASSERT_EQ(constgen_node.at<loco::DataType::S32>(5), -5);
}

TEST(MaxPool2DTest, constructor)
{
  loco::MaxPool2D maxpool_node;

  ASSERT_EQ(maxpool_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(maxpool_node.opcode(), loco::CanonicalOpcode::MaxPool2D);

  ASSERT_EQ(maxpool_node.ifm(), nullptr);

  ASSERT_EQ(maxpool_node.pad()->top(), 0);
  ASSERT_EQ(maxpool_node.pad()->bottom(), 0);
  ASSERT_EQ(maxpool_node.pad()->left(), 0);
  ASSERT_EQ(maxpool_node.pad()->right(), 0);

  ASSERT_EQ(maxpool_node.window()->vertical(), 1);
  ASSERT_EQ(maxpool_node.window()->horizontal(), 1);

  ASSERT_EQ(maxpool_node.stride()->vertical(), 1);
  ASSERT_EQ(maxpool_node.stride()->horizontal(), 1);
}

TEST(MaxPool2DTest, pad)
{
  const uint32_t t = 1;
  const uint32_t b = 2;
  const uint32_t l = 3;
  const uint32_t r = 4;

  loco::MaxPool2D maxpool_node;

  maxpool_node.pad()->top(t);
  ASSERT_EQ(maxpool_node.pad()->top(), t);

  maxpool_node.pad()->bottom(b);
  ASSERT_EQ(maxpool_node.pad()->bottom(), b);

  maxpool_node.pad()->left(l);
  ASSERT_EQ(maxpool_node.pad()->left(), l);

  maxpool_node.pad()->right(r);
  ASSERT_EQ(maxpool_node.pad()->right(), r);
}

TEST(AvgPool2DTest, constructor)
{
  loco::AvgPool2D avgpool_node;

  ASSERT_EQ(avgpool_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(avgpool_node.opcode(), loco::CanonicalOpcode::AvgPool2D);

  ASSERT_EQ(avgpool_node.ifm(), nullptr);

  ASSERT_EQ(avgpool_node.convention(), loco::AvgPool2D::Convention::Unknown);

  ASSERT_EQ(avgpool_node.pad()->top(), 0);
  ASSERT_EQ(avgpool_node.pad()->bottom(), 0);
  ASSERT_EQ(avgpool_node.pad()->left(), 0);
  ASSERT_EQ(avgpool_node.pad()->right(), 0);

  ASSERT_EQ(avgpool_node.window()->vertical(), 1);
  ASSERT_EQ(avgpool_node.window()->horizontal(), 1);

  ASSERT_EQ(avgpool_node.stride()->vertical(), 1);
  ASSERT_EQ(avgpool_node.stride()->horizontal(), 1);
}

TEST(FeatureEncodeTest, constructor)
{
  loco::FeatureEncode feature_encode;

  ASSERT_EQ(feature_encode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(feature_encode.opcode(), loco::CanonicalOpcode::FeatureEncode);

  ASSERT_EQ(feature_encode.input(), nullptr);
  ASSERT_EQ(feature_encode.encoder(), nullptr);
}

TEST(FeatureDecodeTest, constructor)
{
  loco::FeatureDecode feature_decode;

  ASSERT_EQ(feature_decode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(feature_decode.opcode(), loco::CanonicalOpcode::FeatureDecode);

  ASSERT_EQ(feature_decode.input(), nullptr);
  ASSERT_EQ(feature_decode.decoder(), nullptr);
}

TEST(Reshape_Fixed_Test, constructor)
{
  loco::Reshape<loco::ReshapeType::Fixed> reshape;

  ASSERT_EQ(reshape.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(reshape.opcode(), loco::CanonicalOpcode::FixedReshape);

  ASSERT_EQ(reshape.rank(), 0);
}

TEST(Reshape_Fixed_Test, shape)
{
  loco::Reshape<loco::ReshapeType::Fixed> reshape;
  reshape.shape({2, 3});

  ASSERT_EQ(reshape.rank(), 2);
  ASSERT_EQ(reshape.dim(0), 2);
  ASSERT_EQ(reshape.dim(1), 3);
}

TEST(FilterEncodeTest, constructor)
{
  loco::FilterEncode filter_encode;

  ASSERT_EQ(filter_encode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(filter_encode.opcode(), loco::CanonicalOpcode::FilterEncode);

  ASSERT_EQ(filter_encode.input(), nullptr);
  ASSERT_EQ(filter_encode.encoder(), nullptr);
}

TEST(FilterDecodeTest, constructor)
{
  loco::FilterDecode filter_decode;

  ASSERT_EQ(filter_decode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(filter_decode.opcode(), loco::CanonicalOpcode::FilterDecode);

  ASSERT_EQ(filter_decode.input(), nullptr);
  ASSERT_EQ(filter_decode.decoder(), nullptr);
}

TEST(DepthwiseFilterEncodeTest, constructor)
{
  loco::DepthwiseFilterEncode dw_filter_encode;

  ASSERT_EQ(dw_filter_encode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(dw_filter_encode.opcode(), loco::CanonicalOpcode::DepthwiseFilterEncode);

  ASSERT_EQ(dw_filter_encode.input(), nullptr);
  ASSERT_EQ(dw_filter_encode.encoder(), nullptr);
}

TEST(DepthwiseFilterDecodeTest, constructor)
{
  loco::DepthwiseFilterDecode dw_filter_decode;

  ASSERT_EQ(dw_filter_decode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(dw_filter_decode.opcode(), loco::CanonicalOpcode::DepthwiseFilterDecode);

  ASSERT_EQ(dw_filter_decode.input(), nullptr);
  ASSERT_EQ(dw_filter_decode.decoder(), nullptr);
}

TEST(TensorConcatTest, constructor)
{
  loco::TensorConcat tensor_concat;

  ASSERT_EQ(tensor_concat.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(tensor_concat.opcode(), loco::CanonicalOpcode::TensorConcat);

  ASSERT_EQ(tensor_concat.lhs(), nullptr);
  ASSERT_EQ(tensor_concat.rhs(), nullptr);
  ASSERT_EQ(tensor_concat.axis(), 0);

  tensor_concat.axis(3);
  ASSERT_EQ(tensor_concat.axis(), 3);
}

TEST(Conv2DTest, constructor)
{
  loco::Conv2D conv2d;

  ASSERT_EQ(conv2d.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(conv2d.opcode(), loco::CanonicalOpcode::Conv2D);

  ASSERT_EQ(conv2d.ifm(), nullptr);
  ASSERT_EQ(conv2d.ker(), nullptr);

  ASSERT_NE(conv2d.pad(), nullptr);
  ASSERT_EQ(conv2d.pad()->top(), 0);
  ASSERT_EQ(conv2d.pad()->bottom(), 0);
  ASSERT_EQ(conv2d.pad()->left(), 0);
  ASSERT_EQ(conv2d.pad()->right(), 0);

  ASSERT_NE(conv2d.stride(), nullptr);
  ASSERT_EQ(conv2d.stride()->vertical(), 1);
  ASSERT_EQ(conv2d.stride()->horizontal(), 1);
}

TEST(DepthwiseConv2DTest, constructor)
{
  loco::DepthwiseConv2D dw_conv2d;

  ASSERT_EQ(dw_conv2d.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(dw_conv2d.opcode(), loco::CanonicalOpcode::DepthwiseConv2D);

  ASSERT_EQ(dw_conv2d.ifm(), nullptr);
  ASSERT_EQ(dw_conv2d.ker(), nullptr);

  ASSERT_NE(dw_conv2d.pad(), nullptr);
  ASSERT_EQ(dw_conv2d.pad()->top(), 0);
  ASSERT_EQ(dw_conv2d.pad()->bottom(), 0);
  ASSERT_EQ(dw_conv2d.pad()->left(), 0);
  ASSERT_EQ(dw_conv2d.pad()->right(), 0);

  ASSERT_NE(dw_conv2d.stride(), nullptr);
  ASSERT_EQ(dw_conv2d.stride()->vertical(), 1);
  ASSERT_EQ(dw_conv2d.stride()->horizontal(), 1);
}

TEST(TransposedConv2DTest, constructor)
{
  loco::TransposedConv2D tr_conv2d;

  ASSERT_EQ(tr_conv2d.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(tr_conv2d.opcode(), loco::CanonicalOpcode::TransposedConv2D);

  ASSERT_EQ(tr_conv2d.ifm(), nullptr);
  ASSERT_EQ(tr_conv2d.ker(), nullptr);

  ASSERT_NE(tr_conv2d.pad(), nullptr);
  ASSERT_EQ(tr_conv2d.pad()->top(), 0);
  ASSERT_EQ(tr_conv2d.pad()->bottom(), 0);
  ASSERT_EQ(tr_conv2d.pad()->left(), 0);
  ASSERT_EQ(tr_conv2d.pad()->right(), 0);

  ASSERT_NE(tr_conv2d.stride(), nullptr);
  ASSERT_EQ(tr_conv2d.stride()->vertical(), 1);
  ASSERT_EQ(tr_conv2d.stride()->horizontal(), 1);
}

TEST(BiasEncodeTest, constructor)
{
  loco::BiasEncode bias_encode;

  ASSERT_EQ(bias_encode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(bias_encode.opcode(), loco::CanonicalOpcode::BiasEncode);

  ASSERT_EQ(bias_encode.input(), nullptr);
}

TEST(TensorBiasAddTest, constructor)
{
  loco::BiasAdd<loco::Domain::Tensor> bias_add;

  ASSERT_EQ(bias_add.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(bias_add.opcode(), loco::CanonicalOpcode::TensorBiasAdd);

  ASSERT_EQ(bias_add.value(), nullptr);
  ASSERT_EQ(bias_add.bias(), nullptr);
  ASSERT_EQ(bias_add.axis(), 0);
}

TEST(TensorBiasAddTest, alias)
{
  loco::TensorBiasAdd bias_add;

  SUCCEED();
}

TEST(FeatureBiasAddTest, constructor)
{
  loco::BiasAdd<loco::Domain::Feature> bias_add;

  ASSERT_EQ(bias_add.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(bias_add.opcode(), loco::CanonicalOpcode::FeatureBiasAdd);

  ASSERT_EQ(bias_add.value(), nullptr);
  ASSERT_EQ(bias_add.bias(), nullptr);
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

  ASSERT_EQ(sqrt_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(sqrt_node.opcode(), loco::CanonicalOpcode::EltwiseSqrt);

  ASSERT_EQ(sqrt_node.input(), nullptr);
}

TEST(TensorBroadcastTest, constructor)
{
  loco::TensorBroadcast tensor_broadcast_node;

  ASSERT_EQ(tensor_broadcast_node.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(tensor_broadcast_node.opcode(), loco::CanonicalOpcode::TensorBroadcast);

  ASSERT_EQ(tensor_broadcast_node.input(), nullptr);
}

TEST(TensorBroadcastTest, mapping)
{
  loco::TensorBroadcast tensor_broadcast_node;

  ASSERT_EQ(tensor_broadcast_node.mapping()->defined(0), false);

  tensor_broadcast_node.mapping()->dim(0) = 3;

  ASSERT_EQ(tensor_broadcast_node.mapping()->defined(0), true);
  ASSERT_EQ(tensor_broadcast_node.mapping()->dim(0), 3);
}

TEST(MatrixEncodeTest, constructor)
{
  loco::MatrixEncode matrix_encode;

  ASSERT_EQ(matrix_encode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(matrix_encode.opcode(), loco::CanonicalOpcode::MatrixEncode);

  ASSERT_EQ(matrix_encode.input(), nullptr);
}

TEST(MatrixDecodeTest, constructor)
{
  loco::MatrixDecode matrix_decode;

  ASSERT_EQ(matrix_decode.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(matrix_decode.opcode(), loco::CanonicalOpcode::MatrixDecode);

  ASSERT_EQ(matrix_decode.input(), nullptr);
}

TEST(MatMulTest, constructor)
{
  loco::MatMul mat_mul;

  ASSERT_EQ(mat_mul.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(mat_mul.opcode(), loco::CanonicalOpcode::MatMul);

  ASSERT_EQ(mat_mul.lhs(), nullptr);
  ASSERT_EQ(mat_mul.rhs(), nullptr);
}

TEST(TransposeTest, constructor)
{
  loco::TensorTranspose transpose;

  ASSERT_EQ(transpose.dialect(), loco::CanonicalDialect::get());
  ASSERT_EQ(transpose.opcode(), loco::CanonicalOpcode::TensorTranspose);

  ASSERT_EQ(transpose.input(), nullptr);
  ASSERT_EQ(transpose.perm()->size(), 0);
}

TEST(TransposeTest, perm)
{
  loco::TensorTranspose transpose;

  transpose.perm()->size(3);
  transpose.perm()->axis(0) = 1;
  transpose.perm()->axis(1) = 2;
  transpose.perm()->axis(2) = 0;

  ASSERT_EQ(transpose.perm()->axis(0), 1);
  ASSERT_EQ(transpose.perm()->axis(1), 2);
  ASSERT_EQ(transpose.perm()->axis(2), 0);
}
