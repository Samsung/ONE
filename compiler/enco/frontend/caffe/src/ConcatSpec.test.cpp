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

#include "ConcatSpec.h"

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Shape;

namespace
{
class ConcatSpecTest : public ::testing::Test
{
  // FOR FUTURE USE
};
} // namespace

TEST_F(ConcatSpecTest, ifm_shape)
{
  const Shape in_1{1, 1, 4, 4};
  const Shape in_2{1, 2, 4, 4};
  const Shape in_3{1, 3, 4, 4};
  const Shape in_4{1, 4, 4, 4};

  auto expected = Shape{1, 10, 4, 4};
  auto obtained = concat_spec(1).forward({in_1, in_2, in_3, in_4});

  ASSERT_EQ(expected, obtained);
}
