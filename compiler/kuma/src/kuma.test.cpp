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

#include "kuma.h"

#include <gtest/gtest.h>

using namespace kuma;

TEST(GreedyAlgorithmTests, empty)
{
  struct ContextImpl : public Context<Algorithm::Greedy>
  {
    uint32_t item_count(void) const final { return 0; }
    ItemSize item_size(const ItemID &) const final { throw std::runtime_error{"error"}; }

    void mem_offset(const ItemID &, const MemoryOffset &) { throw std::runtime_error{"error"}; };
    void mem_total(const MemorySize &total) final { _total = total; }

    uint32_t _total = 0xffffffff;
  };

  ContextImpl ctx;

  solve(&ctx);

  ASSERT_EQ(ctx._total, 0);
}

TEST(LinearScanFirstFitTests, reuse)
{
  struct ContextImpl : public Context<Algorithm::LinearScanFirstFit>
  {
    uint32_t item_count(void) const final { return 3; }
    ItemSize item_size(const ItemID &) const final { return 4; }

    std::set<ItemID> conflict_with(const ItemID &id) const
    {
      // 0 <-> 1 <-> 2
      switch (id)
      {
        case 0:
          return std::set<ItemID>({1});
        case 1:
          return std::set<ItemID>({0, 2});
        case 2:
          return std::set<ItemID>({1});
        default:
          break;
      };

      throw std::runtime_error{"Invalid"};
    }

    void mem_offset(const ItemID &id, const MemoryOffset &offset) { _offsets[id] = offset; };
    void mem_total(const MemorySize &total) final { _total = total; }

    uint32_t _offsets[3];
    uint32_t _total = 0xffffffff;
  };

  ContextImpl ctx;

  solve(&ctx);

  // EXPECTED MEMORY LAYOUT:
  // ------------------ 0
  // | ITEM 0, ITEM 2 |
  // ------------------ 4
  // | ITEM 1         |
  // ------------------ 8
  ASSERT_EQ(ctx._total, 8);
  ASSERT_EQ(ctx._offsets[0], 0);
  ASSERT_EQ(ctx._offsets[1], 4);
  ASSERT_EQ(ctx._offsets[2], 0);
}
