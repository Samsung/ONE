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

#include "loco/IR/NodeShape.h"

#include <gtest/gtest.h>

TEST(NodeShapeTest, default_constructor)
{
  loco::NodeShape node_shape;

  ASSERT_EQ(node_shape.domain(), loco::Domain::Unknown);
}

TEST(NodeShapeTest, bias_shape_constructor)
{
  loco::BiasShape bias_shape;

  bias_shape.length() = 4;

  loco::NodeShape node_shape{bias_shape};

  ASSERT_EQ(node_shape.domain(), loco::Domain::Bias);
  ASSERT_EQ(node_shape.as<loco::BiasShape>().length(), 4);
}

TEST(NodeShapeTest, dwfilter_shape_constructor)
{
  loco::DepthwiseFilterShape dwfilter_shape;

  dwfilter_shape.depth() = 2;
  dwfilter_shape.multiplier() = 3;
  dwfilter_shape.height() = 4;
  dwfilter_shape.width() = 5;

  loco::NodeShape node_shape{dwfilter_shape};

  ASSERT_EQ(node_shape.domain(), loco::Domain::DepthwiseFilter);
  ASSERT_EQ(node_shape.as<loco::DepthwiseFilterShape>().depth(), 2);
  ASSERT_EQ(node_shape.as<loco::DepthwiseFilterShape>().multiplier(), 3);
  ASSERT_EQ(node_shape.as<loco::DepthwiseFilterShape>().height(), 4);
  ASSERT_EQ(node_shape.as<loco::DepthwiseFilterShape>().width(), 5);
}

TEST(NodeShapeTest, feature_shape_constructor)
{
  loco::FeatureShape feature_shape;

  feature_shape.count() = 2;
  feature_shape.depth() = 3;
  feature_shape.height() = 4;
  feature_shape.width() = 5;

  loco::NodeShape node_shape{feature_shape};

  ASSERT_EQ(node_shape.domain(), loco::Domain::Feature);
  ASSERT_EQ(node_shape.as<loco::FeatureShape>().count(), 2);
  ASSERT_EQ(node_shape.as<loco::FeatureShape>().depth(), 3);
  ASSERT_EQ(node_shape.as<loco::FeatureShape>().height(), 4);
  ASSERT_EQ(node_shape.as<loco::FeatureShape>().width(), 5);
}

TEST(NodeShapeTest, filter_shape_constructor)
{
  loco::FilterShape filter_shape;

  filter_shape.count() = 2;
  filter_shape.depth() = 3;
  filter_shape.height() = 4;
  filter_shape.width() = 5;

  loco::NodeShape node_shape{filter_shape};

  ASSERT_EQ(node_shape.domain(), loco::Domain::Filter);
  ASSERT_EQ(node_shape.as<loco::FilterShape>().count(), 2);
  ASSERT_EQ(node_shape.as<loco::FilterShape>().depth(), 3);
  ASSERT_EQ(node_shape.as<loco::FilterShape>().height(), 4);
  ASSERT_EQ(node_shape.as<loco::FilterShape>().width(), 5);
}

TEST(NodeShapeTest, tensor_shape_constructor)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);
  tensor_shape.dim(0) = 4;
  tensor_shape.dim(1) = 5;

  loco::NodeShape node_shape{tensor_shape};

  ASSERT_EQ(node_shape.domain(), loco::Domain::Tensor);
  ASSERT_EQ(node_shape.as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(node_shape.as<loco::TensorShape>().dim(0), 4);
  ASSERT_EQ(node_shape.as<loco::TensorShape>().dim(1), 5);
}

TEST(NodeShapeTest, copy_constructible)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);
  tensor_shape.dim(0) = 4;
  tensor_shape.dim(1) = 5;

  loco::NodeShape orig{tensor_shape};
  loco::NodeShape copy{orig}; // Call Copy Constructor

  ASSERT_EQ(copy.domain(), loco::Domain::Tensor);
  ASSERT_EQ(copy.as<loco::TensorShape>().rank(), 2);
  ASSERT_EQ(copy.as<loco::TensorShape>().dim(0), 4);
  ASSERT_EQ(copy.as<loco::TensorShape>().dim(1), 5);
}
