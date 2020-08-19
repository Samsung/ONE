/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "ir/Layout.h"
#include "util/ShapeInference.h"

using namespace onert::ir;

TEST(ShapeInference, Elementwise)
{
  Shape lhs_shape{1, 299, 299, 3};
  Shape rhs_shape{3};
  auto infered_out_shape = onert::shape_inference::inferEltwiseShape(lhs_shape, rhs_shape);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.dim(0), 1);
  ASSERT_EQ(infered_out_shape.dim(1), 299);
  ASSERT_EQ(infered_out_shape.dim(2), 299);
  ASSERT_EQ(infered_out_shape.dim(3), 3);
}

TEST(ShapeInference, IncorrectElementwise)
{
  Shape lhs_shape{1, 299, 299, 3};
  Shape rhs_shape{5, 3};
  ASSERT_THROW(onert::shape_inference::inferEltwiseShape(lhs_shape, rhs_shape), std::runtime_error);
}

TEST(ShapeInference, Pool2DNodeSame)
{
  Shape in_shape{10, 6, 12, 20};
  Stride stride{3, 7};
  Padding padding{PaddingType::SAME};

  operation::Pool2D::Param avg_pool_param{
      operation::Pool2D::PoolType::AVG, 3, 6, stride, padding, Activation::NONE};
  auto infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, avg_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::Pool2D::Param max_pool_param{
      operation::Pool2D::PoolType::MAX, 3, 6, stride, padding, Activation::NONE};
  infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, max_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);
}

TEST(ShapeInference, Pool2DNodeValid)
{
  Shape in_shape{10, 6, 12, 20};
  Stride stride{3, 7};
  Padding padding{PaddingType::VALID};

  operation::Pool2D::Param avg_pool_param{
      operation::Pool2D::PoolType::AVG, 3, 6, stride, padding, Activation::NONE};
  auto infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, avg_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::Pool2D::Param max_pool_param{
      operation::Pool2D::PoolType::MAX, 3, 6, stride, padding, Activation::NONE};
  infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, max_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);
}

TEST(ShapeInference, Pool2DNodeExplicit)
{
  Shape in_shape{10, 3, 5, 20};

  Stride stride{3, 7};
  Padding padding{4, 3, 2, 1};

  operation::Pool2D::Param avg_pool_param{
      operation::Pool2D::PoolType::AVG, 3, 6, stride, padding, Activation::NONE};
  auto infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, avg_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::Pool2D::Param max_pool_param{
      operation::Pool2D::PoolType::MAX, 3, 6, stride, padding, Activation::NONE};
  infered_out_shape = onert::shape_inference::inferPoolShape(in_shape, max_pool_param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);
}

TEST(ShapeInference, Conv2D)
{
  Shape in_shape{10, 6, 12, 20};
  Shape ker_shape{30, 3, 6, 20};

  operation::Conv2D::Param param{Stride{3, 7}, Padding{PaddingType::VALID}, Activation::NONE,
                                 Dilation{1, 1}};
  auto infered_out_shape = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 30);

  param = operation::Conv2D::Param{Stride{3, 7}, Padding{PaddingType::SAME}, Activation::NONE,
                                   Dilation{1, 1}};
  infered_out_shape = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 30);

  param =
      operation::Conv2D::Param{Stride{3, 7}, Padding{4, 3, 2, 1}, Activation::NONE, Dilation{1, 1}};
  infered_out_shape = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 3);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 30);
}

TEST(ShapeInference, DepthwiseConv2D)
{
  Shape in_shape{10, 6, 12, 20};
  Shape ker_shape{1, 3, 6, 60};

  operation::DepthwiseConv2D::Param param{Stride{3, 7}, Padding{PaddingType::VALID}, 3,
                                          Activation::NONE};
  auto infered_out_shape =
      onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);

  param = operation::DepthwiseConv2D::Param{Stride{3, 7}, Padding{PaddingType::SAME}, 3,
                                            Activation::NONE};
  infered_out_shape = onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);

  param = operation::DepthwiseConv2D::Param{Stride{3, 7}, Padding{4, 3, 2, 1}, 3, Activation::NONE};
  infered_out_shape = onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 3);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);
}

TEST(ShapeInference, Concat)
{
  {
    Shape in1{10, 20, 30, 3, 50};
    Shape in2{10, 20, 30, 2, 50};
    Shape in3{10, 20, 30, 2, 50};

    operation::Concat::Param param{3};
    auto infered_out_shape = onert::shape_inference::inferConcatShape({in1, in2, in3}, param);

    ASSERT_EQ(infered_out_shape.rank(), 5);
    ASSERT_EQ(infered_out_shape.dim(0), 10);
    ASSERT_EQ(infered_out_shape.dim(1), 20);
    ASSERT_EQ(infered_out_shape.dim(2), 30);
    ASSERT_EQ(infered_out_shape.dim(3), 7);
    ASSERT_EQ(infered_out_shape.dim(4), 50);
  }
  {
    // case 1. when axis < 0
    Shape in1{10, 20, 2};
    Shape in2{10, 20, 3};

    operation::Concat::Param param{-1};
    auto infered_out_shape = onert::shape_inference::inferConcatShape({in1, in2}, param);

    ASSERT_EQ(infered_out_shape.rank(), 3);
    ASSERT_EQ(infered_out_shape.dim(0), 10);
    ASSERT_EQ(infered_out_shape.dim(1), 20);
    ASSERT_EQ(infered_out_shape.dim(2), 5);
  }
  {
    // case 2. when axis < 0
    Shape in1{2, 20, 2};
    Shape in2{3, 20, 2};

    operation::Concat::Param param{-3};
    auto infered_out_shape = onert::shape_inference::inferConcatShape({in1, in2}, param);

    ASSERT_EQ(infered_out_shape.rank(), 3);
    ASSERT_EQ(infered_out_shape.dim(0), 5);
    ASSERT_EQ(infered_out_shape.dim(1), 20);
    ASSERT_EQ(infered_out_shape.dim(2), 2);
  }
}

TEST(ShapeInference, neg_Concat)
{
  {
    operation::Concat::Param param{2};
    Shape in1{10, 1, 3};
    Shape in2{10, 2, 4}; // dim[1] should be 1 but 2

    EXPECT_ANY_THROW(onert::shape_inference::inferConcatShape({in1, in2}, param));
  }
  { // wrong rank
    operation::Concat::Param param{2};
    Shape in1{10, 2, 3, 4};
    Shape in2{10, 2, 4}; // rank should be 4

    EXPECT_ANY_THROW(onert::shape_inference::inferConcatShape({in1, in2}, param));
  }
}

TEST(ShapeInference, ExpandDims)
{
  Shape in_shape{30, 40};

  auto check = [&](int32_t axis, Shape &expected) {
    auto actual = onert::shape_inference::inferExpandDimsShape(in_shape, axis);

    ASSERT_EQ(actual.rank(), 3);
    for (int32_t dim = 0; dim < expected.rank(); dim++)
      ASSERT_EQ(actual.dim(dim), expected.dim(dim));
  };

  { // boundary
    int32_t axis = 0;
    Shape expected{1, 30, 40};
    check(axis, expected);
  }
  { // boundary
    int32_t axis = 2;
    Shape expected{30, 40, 1};
    check(axis, expected);
  }
  { // inside
    int32_t axis = 1;
    Shape expected{30, 1, 40};
    check(axis, expected);
  }
  { // negative boundary
    int32_t axis = -1;
    Shape expected{30, 40, 1};
    check(axis, expected);
  }
  { // negative boundary
    int32_t axis = -3;
    Shape expected{1, 30, 40};
    check(axis, expected);
  }
}

TEST(ShapeInference, neg_ExpandDims)
{
  Shape in_shape{30, 40};

  { // over boundary
    int32_t axis = 3;
    ASSERT_THROW(onert::shape_inference::inferExpandDimsShape(in_shape, axis), std::runtime_error);
  }
  { // over boundary
    int32_t axis = -4;
    ASSERT_THROW(onert::shape_inference::inferExpandDimsShape(in_shape, axis), std::runtime_error);
  }
}

TEST(ShapeInference, FullyConnected)
{
  Shape in_shape{3, 4, 5, 6};
  Shape ker_shape{3, 10};
  auto infered_out_shape = onert::shape_inference::inferFullyConnectedShape(in_shape, ker_shape);

  ASSERT_EQ(infered_out_shape.rank(), 2);
  ASSERT_EQ(infered_out_shape.dim(0), 36);
  ASSERT_EQ(infered_out_shape.dim(1), 3);
}

TEST(ShapeInference, Transpose)
{
  auto check = [&](Shape &in_shape, std::vector<int> perm, Shape &expected) {
    // pre-conditions
    ASSERT_EQ(in_shape.rank(), perm.size());
    ASSERT_EQ(expected.rank(), perm.size());
    auto inferred_out_shape = onert::shape_inference::inferTransposeShape(in_shape, perm);
    // post-conditions
    ASSERT_EQ(inferred_out_shape.rank(), perm.size());
    for (int32_t dim = 0; dim < expected.rank(); dim++)
    {
      ASSERT_EQ(inferred_out_shape.dim(dim), expected.dim(dim));
    }
  };
  // check for 2-D
  {
    Shape in_shape{2, 3};
    std::vector<int> perm = {1, 0};
    Shape expected{3, 2};
    // int32_t rank = 2;
    check(in_shape, perm, expected);
  }
  // check for 3-D
  {
    Shape in_shape{1, 2, 3};
    std::vector<int> perm = {2, 0, 1};
    Shape expected{3, 1, 2};
    // int32_t rank = 3;
    check(in_shape, perm, expected);
  }
  // check for 4-D
  {
    Shape in_shape{1, 2, 3, 4};
    std::vector<int> perm = {1, 3, 0, 2};
    Shape expected{2, 4, 1, 3};
    // int32_t rank = 4;
    check(in_shape, perm, expected);
  }
}

TEST(ShapeInference, neg_Transpose)
{
  Shape in_shape{1, 2, 3};
  // Invalid parameter size
  {
    std::vector<int> perm = {2, 0, 1, 0};
    // int32_t rank = 3;
    ASSERT_THROW(onert::shape_inference::inferTransposeShape(in_shape, perm), std::runtime_error);
  }
  // Invalid parameter value
  {
    std::vector<int> perm = {2, 0, 3};
    // int32_t rank = 3;
    ASSERT_THROW(onert::shape_inference::inferTransposeShape(in_shape, perm), std::runtime_error);
  }
}
