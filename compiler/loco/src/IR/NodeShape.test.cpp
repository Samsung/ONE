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

  ASSERT_EQ(loco::Domain::Unknown, node_shape.domain());
}

TEST(NodeShapeTest, bias_shape_constructor)
{
  loco::BiasShape bias_shape;

  bias_shape.length() = 4;

  loco::NodeShape node_shape{bias_shape};

  ASSERT_EQ(loco::Domain::Bias, node_shape.domain());
  ASSERT_EQ(4, node_shape.as<loco::BiasShape>().length());
}

TEST(NodeShapeTest, dwfilter_shape_constructor)
{
  loco::DepthwiseFilterShape dwfilter_shape;

  dwfilter_shape.depth() = 2;
  dwfilter_shape.multiplier() = 3;
  dwfilter_shape.height() = 4;
  dwfilter_shape.width() = 5;

  loco::NodeShape node_shape{dwfilter_shape};

  ASSERT_EQ(loco::Domain::DepthwiseFilter, node_shape.domain());
  ASSERT_EQ(2, node_shape.as<loco::DepthwiseFilterShape>().depth());
  ASSERT_EQ(3, node_shape.as<loco::DepthwiseFilterShape>().multiplier());
  ASSERT_EQ(4, node_shape.as<loco::DepthwiseFilterShape>().height());
  ASSERT_EQ(5, node_shape.as<loco::DepthwiseFilterShape>().width());
}

TEST(NodeShapeTest, feature_shape_constructor)
{
  loco::FeatureShape feature_shape;

  feature_shape.count() = 2;
  feature_shape.depth() = 3;
  feature_shape.height() = 4;
  feature_shape.width() = 5;

  loco::NodeShape node_shape{feature_shape};

  ASSERT_EQ(loco::Domain::Feature, node_shape.domain());
  ASSERT_EQ(2, node_shape.as<loco::FeatureShape>().count());
  ASSERT_EQ(3, node_shape.as<loco::FeatureShape>().depth());
  ASSERT_EQ(4, node_shape.as<loco::FeatureShape>().height());
  ASSERT_EQ(5, node_shape.as<loco::FeatureShape>().width());
}

TEST(NodeShapeTest, filter_shape_constructor)
{
  loco::FilterShape filter_shape;

  filter_shape.count() = 2;
  filter_shape.depth() = 3;
  filter_shape.height() = 4;
  filter_shape.width() = 5;

  loco::NodeShape node_shape{filter_shape};

  ASSERT_EQ(loco::Domain::Filter, node_shape.domain());
  ASSERT_EQ(2, node_shape.as<loco::FilterShape>().count());
  ASSERT_EQ(3, node_shape.as<loco::FilterShape>().depth());
  ASSERT_EQ(4, node_shape.as<loco::FilterShape>().height());
  ASSERT_EQ(5, node_shape.as<loco::FilterShape>().width());
}

TEST(NodeShapeTest, tensor_shape_constructor)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);
  tensor_shape.dim(0) = 4;
  tensor_shape.dim(1) = 5;

  loco::NodeShape node_shape{tensor_shape};

  ASSERT_EQ(loco::Domain::Tensor, node_shape.domain());
  ASSERT_EQ(2, node_shape.as<loco::TensorShape>().rank());
  ASSERT_EQ(4, node_shape.as<loco::TensorShape>().dim(0));
  ASSERT_EQ(5, node_shape.as<loco::TensorShape>().dim(1));
}

TEST(NodeShapeTest, copy_constructible)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);
  tensor_shape.dim(0) = 4;
  tensor_shape.dim(1) = 5;

  loco::NodeShape orig{tensor_shape};
  loco::NodeShape copy{orig}; // Call Copy Constructor

  ASSERT_EQ(loco::Domain::Tensor, copy.domain());
  ASSERT_EQ(2, copy.as<loco::TensorShape>().rank());
  ASSERT_EQ(4, copy.as<loco::TensorShape>().dim(0));
  ASSERT_EQ(5, copy.as<loco::TensorShape>().dim(1));
}
