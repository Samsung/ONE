/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/Graph.h"
#include "mir/ops/AddOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/ResizeOp.h"
#include "mir/ops/SqueezeOp.h"
#include "mir/ops/ReduceMeanOp.h"
#include "mir/Shape.h"

#include <vector>

#include "gtest/gtest.h"

using namespace mir;

TEST(ShapeInferenceTest, BidirectionalBroadcast)
{
  const Shape shape1{2, 1, 2};
  const Shape shape2{3, 1};
  const Shape reference{2, 3, 2};

  const Shape result1 = broadcastShapes(shape1, shape2);
  const Shape result2 = broadcastShapes(shape2, shape1);

  ASSERT_EQ(result1, reference);
  ASSERT_EQ(result2, reference);
}

TEST(ShapeInferenceTest, ReshapeAutoDimension)
{
  Graph g;

  Shape input_shape{10, 2, 5};
  Shape expected_shape{10, 1, 10};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto op = g.create<ops::ReshapeOp>(input->getOutput(0), Shape{10, 1, Shape::autoDim});

  ASSERT_EQ(expected_shape, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, ResizeWithShape)
{
  Graph g;

  Shape result_shape{2, 10, 10, 3};

  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 5, 5, 3}};
  auto input = g.create<ops::InputOp>(input_type);

  auto op = g.create<ops::ResizeOp>(input->getOutput(0),
                                    ops::ResizeOp::ResizeMethod::nearestNeighbor, result_shape);

  ASSERT_EQ(result_shape, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, ResizeWithScale)
{
  Graph g;

  Shape result_shape{1, 30, 10, 3};

  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 5, 5, 3}};
  auto input = g.create<ops::InputOp>(input_type);

  auto op =
    g.create<ops::ResizeOp>(input->getOutput(0), ops::ResizeOp::ResizeMethod::nearestNeighbor,
                            std::vector<float>{1, 6, 2, 1});

  ASSERT_EQ(result_shape, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, ReduceChangeRank)
{
  Graph g;

  Shape resultShape{10, 10};

  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{10, 2, 10, 9}};
  auto input = g.create<ops::InputOp>(input_type);

  auto n = g.create<ops::ReduceMeanOp>(input->getOutput(0), std::vector<int32_t>{1, 3}, false);

  ASSERT_EQ(resultShape, n->getOutputShape(0));
}

TEST(ShapeInferenceTest, ReshapeAutoDimensionShrink)
{
  Graph g;

  Shape input_shape{10, 2, 10};
  Shape result_shape_shrink{10, 20};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto op = g.create<ops::ReshapeOp>(input->getOutput(0), Shape{10, Shape::autoDim});

  ASSERT_EQ(result_shape_shrink, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, ReshapeAutoDimensionExpand)
{
  Graph g;

  Shape input_shape{10, 2, 10};
  Shape result_shape_expand{5, 10, 2, 2};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto op = g.create<ops::ReshapeOp>(input->getOutput(0), Shape{5, Shape::autoDim, 2, 2});

  ASSERT_EQ(result_shape_expand, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, ReshapeAutoDimensionUnsqueeze)
{
  Graph g;

  Shape input_shape{10, 2, 10};
  Shape result_shape_expand{1, 10, 2, 1, 10, 1};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto op = g.create<ops::ReshapeOp>(input->getOutput(0), Shape{1, Shape::autoDim, 2, 1, 10, 1});

  ASSERT_EQ(result_shape_expand, op->getOutputShape(0));
}

TEST(ShapeInferenceTest, SqueezeTestAllDims)
{
  Graph g;

  Shape input_shape{1, 2, 1, 4};
  Shape expected_shape{2, 4};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto sq1 = g.create<ops::SqueezeOp>(input->getOutput(0), std::vector<int32_t>{});

  ASSERT_EQ(sq1->getOutputShape(0), expected_shape);
}

TEST(ShapeInferenceTest, ElementwiseBC)
{
  Graph g;

  Shape input_shape{1, 10, 10, 1};
  Shape input2_shape{1, 1, 10, 10};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  mir::TensorType input2_type{mir::DataType::FLOAT32, input2_shape};

  auto input = g.create<ops::InputOp>(input_type);
  auto input2 = g.create<ops::InputOp>(input2_type);

  auto add = g.create<ops::AddOp>(input->getOutput(0), input2->getOutput(0));

  ASSERT_EQ(add->getOutputShape(0), Shape({1, 10, 10, 10}));
}

TEST(ShapeInferenceTest, SqueezeTestSpecificDims)
{
  Graph g;

  Shape input_shape{1, 2, 1, 4};
  Shape expected_shape{1, 2, 4};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto sq1 = g.create<ops::SqueezeOp>(input->getOutput(0), std::vector<int32_t>{2});

  ASSERT_EQ(sq1->getOutputShape(0), expected_shape);
}

TEST(ShapeInferenceTest, SqueezeTestScalarResult)
{
  Graph g;

  Shape input_shape{1, 1, 1, 1};
  Shape expected_shape{1};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  auto input = g.create<ops::InputOp>(input_type);
  auto sq1 = g.create<ops::SqueezeOp>(input->getOutput(0), std::vector<int32_t>{});

  ASSERT_EQ(sq1->getOutputShape(0), expected_shape);
}
