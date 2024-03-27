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

#include "backend/basic/MemoryPlanner.h"
#include "ir/Index.h"

namespace
{
using Index = onert::ir::OperandIndex;
}

TEST(Allocator, allocate_test)
{
  ::onert::backend::basic::Allocator allocator(1024);
  ASSERT_NE(allocator.base(), nullptr);
}

TEST(BumpPlanner, claim_test)
{
  ::onert::backend::basic::BumpPlanner<Index> planner;

  auto claim = [&planner](uint32_t index, size_t size, uint32_t expected_offset) {
    Index mem_idx(index);
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  claim(0, 10, 0);
  claim(1, 20, 10);
  claim(2, 30, 30);
}

TEST(FirstFitPlanner, claim_release_test)
{
  ::onert::backend::basic::FirstFitPlanner<Index> planner;

  auto claim = [&planner](uint32_t index, size_t size, uint32_t expected_offset) {
    Index mem_idx(index);
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto release = [&planner](uint32_t index) {
    Index mem_idx(index);
    planner.release(mem_idx);
  };

  // 0 CLAIM - 10
  claim(0, 10, 0);

  // 1 CLAIM - 20
  claim(1, 20, 10);

  // 2 CLAIM - 30
  claim(2, 30, 30);

  // 0 RELEASE - 10
  release(0);

  // 3 CLAIM - 20
  claim(3, 20, 60);

  // 4 CLAIM - 5
  claim(4, 5, 0);

  // 5 CLAIM - 10
  claim(5, 10, 80);

  // 6 CLAIM - 5
  claim(6, 5, 5);

  // 2 RELEASE - 30
  release(2);

  // 7 CLAIM - 35
  claim(7, 35, 90);

  // 8 CLAIM - 10
  claim(8, 10, 30);

  // 4 RELEASE - 5
  release(4);

  // 9 CLAIM - 10
  claim(9, 10, 40);

  // 10 CLAIM - 10
  claim(10, 10, 50);

  // 6 RELEASE
  release(6);

  // 1 RELEASE
  release(1);

  // 8 RELEASE
  release(8);

  // 9 RELEASE
  release(9);

  // 10 RELEASE
  release(10);

  // 3 RELEASE
  release(3);

  // 5 RELEASE
  release(5);

  // 7 RELEASE
  release(7);
}

TEST(WICPlanner, claim_release_test)
{
  ::onert::backend::basic::WICPlanner<Index> planner;

  auto claim = [&planner](uint32_t index, size_t size) {
    Index mem_idx(index);
    planner.claim(mem_idx, size);
  };

  auto release = [&planner](uint32_t index) {
    Index mem_idx(index);
    planner.release(mem_idx);
  };

  auto verify = [&planner](uint32_t index, uint32_t size, uint32_t expected_offset) {
    Index mem_idx(index);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto capacity = [&planner](uint32_t expected_capacity) {
    auto actual_capacity = planner.capacity();
    ASSERT_EQ(actual_capacity, expected_capacity);
  };

  claim(0, 20);
  claim(1, 5);
  release(0);
  claim(2, 10);
  release(1);
  claim(3, 10);
  release(2);
  claim(4, 10);
  release(3);
  claim(5, 20);
  release(4);
  claim(6, 20);
  release(5);
  release(7);

  // VERIFY 0 - 0
  verify(0, 20, 0);

  // VERIFY 1 - 20
  verify(1, 5, 20);

  // VERIFY 2 - 0
  verify(2, 10, 0);

  // VERIFY 3 - 10
  verify(3, 10, 10);

  // VERIFY 4 - 20
  verify(4, 10, 20);

  // VERIFY 5 - 0
  verify(5, 20, 0);

  // VERIFY 6 - 20
  verify(6, 20, 20);

  // CAPACITY - 40
  capacity(40);
}
