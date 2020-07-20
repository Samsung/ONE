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

#include "FuseInstanceNormPassInternal.h"

#include <vector>

#include <gtest/gtest.h>

namespace
{

void setShape(luci::CircleNode &node, const std::vector<int> &v)
{
  node.rank(v.size());
  for (int i = 0; i < v.size(); ++i)
  {
    node.dim(i) = v[i];
  }
}

} // namespace

TEST(FuseInstanceNormPass, is_quasi_1D_with_dummy_dim)
{
  luci::CircleConst const_node;

  setShape(const_node, {});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {1});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {8});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {1, 2, 1, 8, 1});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {8, 3});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {8, 1});
  EXPECT_FALSE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {1, 8, 1});
  EXPECT_TRUE(is_quasi_1D_with_dummy_dim(&const_node, 8));

  setShape(const_node, {1, 1, 1, 8, 1});
  EXPECT_TRUE(is_quasi_1D_with_dummy_dim(&const_node, 8));
}
