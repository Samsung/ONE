/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MinMaxComputer.h"

#include <luci/IR/CircleNodes.h>

#include <unordered_map>

#include <gtest/gtest.h>

using namespace record_minmax;

TEST(MinMaxComputerTest, percentile)
{
  auto computer = make_percentile_computer(0.0, 100.0);

  luci::CircleAdd node;
  MinMaxVectors minmax;
  {
    minmax.min_vector = {1.0, 2.0, 3.0};
    minmax.max_vector = {4.0, 5.0, 6.0};
  }
  std::unordered_map<const luci::CircleNode *, MinMaxVectors> min_max_map;
  min_max_map.insert({&node, minmax});

  computer->update_qparam(&min_max_map);

  EXPECT_TRUE(node.quantparam() != nullptr);
}

TEST(MinMaxComputerTest, percentile_nullptr_NEG)
{
  auto computer = make_percentile_computer(0.0, 100.0);

  EXPECT_ANY_THROW(computer->update_qparam(nullptr));
}

TEST(MinMaxComputerTest, moving_avg)
{
  auto computer = make_moving_avg_computer(1, 0.99);

  luci::CircleAdd node;
  MinMaxVectors minmax;
  {
    minmax.min_vector = {1.0, 2.0, 3.0};
    minmax.max_vector = {4.0, 5.0, 6.0};
  }
  std::unordered_map<const luci::CircleNode *, MinMaxVectors> min_max_map;
  min_max_map.insert({&node, minmax});

  computer->update_qparam(&min_max_map);

  EXPECT_TRUE(node.quantparam() != nullptr);
}

TEST(MinMaxComputerTest, moving_avg_nullptr_NEG)
{
  auto computer = make_moving_avg_computer(1, 0.99);

  EXPECT_ANY_THROW(computer->update_qparam(nullptr));
}
