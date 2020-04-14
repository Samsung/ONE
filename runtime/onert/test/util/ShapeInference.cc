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
  auto infered_shapes = onert::shape_inference::inferEltwiseShape(lhs_shape, rhs_shape);
  auto infered_out_shape = infered_shapes[0];

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

  operation::AvgPool2D::Param avg_pool_param{3, 6, stride, padding, Activation::NONE};
  auto infered_shapes = onert::shape_inference::inferAvgPoolShape(in_shape, avg_pool_param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::MaxPool2D::Param max_pool_param{3, 6, stride, padding, Activation::NONE};
  infered_shapes = onert::shape_inference::inferMaxPoolShape(in_shape, max_pool_param);
  infered_out_shape = infered_shapes[0];

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

  operation::AvgPool2D::Param avg_pool_param{3, 6, stride, padding, Activation::NONE};
  auto infered_shapes = onert::shape_inference::inferAvgPoolShape(in_shape, avg_pool_param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::MaxPool2D::Param max_pool_param{3, 6, stride, padding, Activation::NONE};
  infered_shapes = onert::shape_inference::inferMaxPoolShape(in_shape, max_pool_param);
  infered_out_shape = infered_shapes[0];

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

  operation::AvgPool2D::Param avg_pool_param{3, 6, stride, padding, Activation::NONE};
  auto infered_shapes = onert::shape_inference::inferAvgPoolShape(in_shape, avg_pool_param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 20);

  operation::MaxPool2D::Param max_pool_param{3, 6, stride, padding, Activation::NONE};
  infered_shapes = onert::shape_inference::inferMaxPoolShape(in_shape, max_pool_param);
  infered_out_shape = infered_shapes[0];

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

  operation::Conv2D::Param param{Stride{3, 7}, Padding{PaddingType::VALID}, Activation::NONE};
  auto infered_shapes = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 30);

  param = operation::Conv2D::Param{Stride{3, 7}, Padding{PaddingType::SAME}, Activation::NONE};
  infered_shapes = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);
  infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 30);

  param = operation::Conv2D::Param{Stride{3, 7}, Padding{4, 3, 2, 1}, Activation::NONE};
  infered_shapes = onert::shape_inference::inferConv2DShape(in_shape, ker_shape, param);
  infered_out_shape = infered_shapes[0];

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
  auto infered_shapes =
      onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 1);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);

  param = operation::DepthwiseConv2D::Param{Stride{3, 7}, Padding{PaddingType::SAME}, 3,
                                            Activation::NONE};
  infered_shapes = onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);
  infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);

  param = operation::DepthwiseConv2D::Param{Stride{3, 7}, Padding{4, 3, 2, 1}, 3, Activation::NONE};
  infered_shapes = onert::shape_inference::inferDepthwiseConv2DShape(in_shape, ker_shape, param);
  infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 4);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).N, 10);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).H, 3);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).W, 2);
  ASSERT_EQ(infered_out_shape.asFeature(Layout::NHWC).C, 60);
}

TEST(ShapeInference, Concat)
{
  Shape in1{10, 20, 30, 3, 50};
  Shape in2{10, 20, 30, 2, 50};
  Shape in3{10, 20, 30, 2, 50};

  operation::Concat::Param param{3};
  auto infered_shapes = onert::shape_inference::inferConcatShape({in1, in2, in3}, param);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 5);
  ASSERT_EQ(infered_out_shape.dim(0), 10);
  ASSERT_EQ(infered_out_shape.dim(1), 20);
  ASSERT_EQ(infered_out_shape.dim(2), 30);
  ASSERT_EQ(infered_out_shape.dim(3), 7);
  ASSERT_EQ(infered_out_shape.dim(4), 50);
}

TEST(ShapeInference, FullyConnected)
{
  Shape in_shape{3, 4, 5, 6};
  Shape ker_shape{3, 10};
  auto infered_shapes = onert::shape_inference::inferFullyConnectedShape(in_shape, ker_shape);
  auto infered_out_shape = infered_shapes[0];

  ASSERT_EQ(infered_out_shape.rank(), 2);
  ASSERT_EQ(infered_out_shape.dim(0), 36);
  ASSERT_EQ(infered_out_shape.dim(1), 3);
}
