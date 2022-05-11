/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include <luci/IR/CircleNodes.h>

#include "InputDataLoader.h"

using namespace circle_eval_diff;

TEST(CircleEvalInputDataLoaderTest, verifyTypeShapeTest)
{
  luci::CircleInput input;
  input.dtype(loco::DataType::FLOAT32);
  input.rank(4);
  input.dim(0).set(1);
  input.dim(1).set(3);
  input.dim(2).set(3);
  input.dim(3).set(2);

  loco::DataType right_data_type{loco::DataType::FLOAT32};
  std::vector<loco::Dimension> right_shape;
  right_shape.emplace_back(1);
  right_shape.emplace_back(3);
  right_shape.emplace_back(3);
  right_shape.emplace_back(2);

  EXPECT_NO_THROW(verifyTypeShape(&input, right_data_type, right_shape));
}

TEST(CircleEvalInputDataLoaderTest, verifyTypeShapeTest_NEG)
{
  luci::CircleInput input;
  input.dtype(loco::DataType::FLOAT32);
  input.rank(4);
  input.dim(0).set(1);
  input.dim(1).set(4);
  input.dim(2).set(4);
  input.dim(3).set(2);

  loco::DataType right_data_type{loco::DataType::FLOAT32};
  loco::DataType wrong_data_type{loco::DataType::FLOAT16};
  std::vector<loco::Dimension> wrong_shape;
  wrong_shape.emplace_back(1);
  wrong_shape.emplace_back(3);
  wrong_shape.emplace_back(3);
  wrong_shape.emplace_back(2);

  EXPECT_ANY_THROW(verifyTypeShape(&input, right_data_type, wrong_shape));
  EXPECT_ANY_THROW(verifyTypeShape(&input, wrong_data_type, wrong_shape));
}
