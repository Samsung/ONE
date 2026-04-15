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

#include "coco/IR/BlockManager.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace
{
class BlockManagerTest : public ::testing::Test
{
public:
  // Create a coco::BlockManager for testing
  coco::BlockManager *allocate(void)
  {
    auto p = new coco::BlockManager;
    _allocated.emplace_back(p);
    return p;
  }

private:
  std::vector<std::unique_ptr<coco::BlockManager>> _allocated;
};
} // namespace

TEST_F(BlockManagerTest, create)
{
  auto mgr = allocate();
  auto blk = mgr->create();

  ASSERT_NE(blk, nullptr);
}

TEST_F(BlockManagerTest, destroy)
{
  auto mgr = allocate();
  auto blk_1 = mgr->create();
  auto blk_2 = mgr->create();

  mgr->destroy(blk_1);

  ASSERT_EQ(mgr->size(), 1);
  ASSERT_EQ(mgr->at(0), blk_2);
}
