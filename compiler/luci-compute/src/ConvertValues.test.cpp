/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertValues.h"

#include <gtest/gtest.h>

TEST(ConvertValues, get_act_minmax)
{
  auto func = luci::compute::FusedActFunc::RELU6;
  float act_min, act_max;
  ASSERT_NO_THROW(luci::compute::get_act_minmax(func, act_min, act_max));
  EXPECT_EQ(act_min, 0);
  EXPECT_EQ(act_max, 6);
}

TEST(ConvertValues, get_act_minmax_NEG)
{
  // force convert with invalid value as future unhandled value
  luci::compute::FusedActFunc func = static_cast<luci::compute::FusedActFunc>(250);
  float act_min, act_max;
  ASSERT_ANY_THROW(luci::compute::get_act_minmax(func, act_min, act_max));
}
