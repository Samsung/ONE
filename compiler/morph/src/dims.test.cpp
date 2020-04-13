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

#include "morph/dims.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

TEST(DimsTest, as_dims_from_tensor)
{
  auto dims = morph::as_dims(tensor::Shape{1, 3, 4, 5});

  ASSERT_EQ(dims.size(), 4);
  ASSERT_EQ(dims.at(0), 1);
  ASSERT_EQ(dims.at(1), 3);
  ASSERT_EQ(dims.at(2), 4);
  ASSERT_EQ(dims.at(3), 5);
}
