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

#include "pepper/strcast.h"

#include <gtest/gtest.h>

TEST(StrCastTests, safe_strcast_int)
{
  ASSERT_EQ(pepper::safe_strcast<int>("0", 128), 0);
  ASSERT_EQ(pepper::safe_strcast<int>("2", 128), 2);
  ASSERT_EQ(pepper::safe_strcast<int>(nullptr, 128), 128);
}
