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

#include "mir/TensorVariant.h"

#include <gtest/gtest.h>

using namespace mir;

TEST(TensorVariant, BasicTest)
{
  Shape shape{2, 2};
  TensorVariant t(DataType::FLOAT32, shape);

  ASSERT_EQ(t.getShape(), shape);
  ASSERT_EQ(t.getOffset({0, 0}), 0u);
}

TEST(TensorVariant, ElementSizeDeductionTest)
{
  Shape shape{2, 2, 2};
  TensorVariant t(DataType::FLOAT32, shape);

  ASSERT_EQ(t.getElementSize(), sizeof(float));
  ASSERT_EQ((float *)t.at({1, 1, 1}), (float *)t.at({0, 0, 0}) + 7);
}
