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

#include "RandomUtils.h"

#include <gtest/gtest.h>

using namespace dalgona;

TEST(DalgonaUtilTest, gen_random_int32)
{
  const uint32_t num_elements = 10;
  const int32_t min = -5;
  const int32_t max = 5;
  std::vector<int32_t> buffer = genRandomIntData<int32_t>(num_elements, min, max);

  EXPECT_EQ(num_elements, buffer.size());
  for (auto val : buffer)
  {
    EXPECT_TRUE(val >= min and val <= max);
  }
}

TEST(DalgonaUtilTest, gen_random_int32_NEG)
{
  const uint32_t num_elements = 10;
  const int32_t min = 5;
  const int32_t max = -5;
  EXPECT_ANY_THROW(genRandomIntData<int32_t>(num_elements, min, max));
}

TEST(DalgonaUtilTest, gen_random_float)
{
  const uint32_t num_elements = 10;
  const float min = -5;
  const float max = 5;
  std::vector<float> buffer = genRandomFloatData(num_elements, min, max);

  EXPECT_EQ(num_elements, buffer.size());
  for (auto val : buffer)
  {
    EXPECT_TRUE(val >= min and val <= max);
  }
}

TEST(DalgonaUtilTest, gen_random_float_NEG)
{
  const uint32_t num_elements = 10;
  const float min = 5;
  const float max = -5;
  EXPECT_ANY_THROW(genRandomFloatData(num_elements, min, max));
}
