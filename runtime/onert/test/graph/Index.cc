/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "util/Index.h"

using Index = ::onert::util::Index<uint32_t, struct TestTag>;

TEST(Index, neg_index_test)
{
  Index idx1{1u};
  Index idx2{2u};
  Index idx3{idx1};

  ASSERT_EQ(idx1, 1);
  ASSERT_EQ(idx1, 1u);
  ASSERT_EQ(idx1.value(), 1u);
  ASSERT_NE(idx1, idx2);
  ASSERT_EQ(idx1, idx3);
}
