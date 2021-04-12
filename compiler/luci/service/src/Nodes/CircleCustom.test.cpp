/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/CircleNodeClone.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(CloneNodeTest, clone_Custom)
{
  auto g = loco::make_graph();
  auto node_custom = g->nodes()->create<luci::CircleCustom>(2, 3);
  std::vector<uint8_t> options({0x55, 0x56, 0x57});
  std::string code = "hello";
  node_custom->custom_options(options);
  node_custom->custom_code(code);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_custom, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_custom = dynamic_cast<luci::CircleCustom *>(cloned);
  ASSERT_NE(nullptr, cloned_custom);
  auto cloned_options = cloned_custom->custom_options();
  ASSERT_EQ(options.size(), cloned_options.size());
  auto size = options.size();
  for (size_t s = 0; s < size; ++s)
    ASSERT_EQ(options.at(s), cloned_options.at(s));
  ASSERT_TRUE(node_custom->custom_code() == cloned_custom->custom_code());
}
