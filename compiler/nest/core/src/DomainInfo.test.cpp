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

#include "nest/DomainInfo.h"

#include <gtest/gtest.h>

TEST(DOMAIN_INFO, ctor)
{
  nest::DomainInfo info{1, 2, 3, 4};

  ASSERT_EQ(4, info.rank());
  ASSERT_EQ(1, info.dim(0));
  ASSERT_EQ(2, info.dim(1));
  ASSERT_EQ(3, info.dim(2));
  ASSERT_EQ(4, info.dim(3));
}
