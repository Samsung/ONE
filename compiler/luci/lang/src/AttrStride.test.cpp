/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/AttrStride.h"

#include <gtest/gtest.h>

TEST(CircleAttrStrideTest, set)
{
  auto s = luci::Stride();

  s.h(10u);
  s.w(10u);

  ASSERT_EQ(s.h(), 10u);
  ASSERT_EQ(s.w(), 10u);

  s.h(10); // int32_t
  s.w(10);

  ASSERT_EQ(s.h(), 10u);
  ASSERT_EQ(s.w(), 10u);
}
