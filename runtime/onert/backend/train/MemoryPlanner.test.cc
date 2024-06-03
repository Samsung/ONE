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

#include "MemoryPlanner.h"
#include "ir/Index.h"

using namespace onert::backend::train;
using onert::ir::OperandIndex;
using onert::ir::OperationIndex;

TEST(BumpPlanner, claim_test)
{
  BumpPlanner planner;

  auto claim = [&planner](uint32_t op_index, uint32_t operand_index, size_t size,
                          uint32_t expected_offset) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  claim(0, 0, 10, 0);
  claim(1, 0, 20, 10);
  claim(2, 2, 30, 30);
}

TEST(FirstFitPlanner, claim_release_test)
{
  FirstFitPlanner planner;

  auto claim = [&planner](uint32_t op_index, uint32_t operand_index, size_t size,
                          uint32_t expected_offset) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto release = [&planner](uint32_t op_index, uint32_t operand_index) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.release(mem_idx);
  };

  // 0 CLAIM - 10
  claim(0, 0, 10, 0);

  // 1 CLAIM - 20
  claim(1, 0, 20, 10);

  // 2 CLAIM - 30
  claim(2, 2, 30, 30);

  // 0 RELEASE - 10
  release(0, 0);

  // 3 CLAIM - 20
  claim(3, 1, 20, 60);

  // 4 CLAIM - 5
  claim(4, 1, 5, 0);

  // 5 CLAIM - 10
  claim(5, 1, 10, 80);

  // 6 CLAIM - 5
  claim(6, 1, 5, 5);

  // 2 RELEASE - 30
  release(2, 2);

  // 7 CLAIM - 35
  claim(7, 1, 35, 90);

  // 8 CLAIM - 10
  claim(8, 1, 10, 30);

  // 4 RELEASE - 5
  release(4, 1);

  // 9 CLAIM - 10
  claim(9, 0, 10, 40);

  // 10 CLAIM - 10
  claim(10, 0, 10, 50);

  // 6 RELEASE
  release(6, 1);

  // 1 RELEASE
  release(1, 0);

  // 8 RELEASE
  release(8, 1);

  // 9 RELEASE
  release(9, 0);

  // 10 RELEASE
  release(10, 0);

  // 3 RELEASE
  release(3, 1);

  // 5 RELEASE
  release(5, 1);

  // 7 RELEASE
  release(7, 1);
}

TEST(FirstFitPlanner, neg_release_non_existing_index)
{
  FirstFitPlanner planner;

  auto claim = [&planner](uint32_t op_index, uint32_t operand_index, size_t size,
                          uint32_t expected_offset) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto release = [&planner](uint32_t op_index, uint32_t operand_index) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.release(mem_idx);
  };

  // 0 CLAIM - 10
  claim(0, 0, 10, 0);

  // 1 CLAIM - 20
  claim(1, 0, 20, 10);

  // 2 CLAIM - 30
  claim(2, 2, 30, 30);

  // RELEASE non-existing index
  auto on_only_debug_mode = [&release]() {
    EXPECT_DEATH({ release(0, 1); },
                 "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };
  assert(on_only_debug_mode());
}

TEST(FirstFitPlanner, neg_release_twice)
{
  FirstFitPlanner planner;

  auto claim = [&planner](uint32_t op_index, uint32_t operand_index, size_t size,
                          uint32_t expected_offset) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.claim(mem_idx, size);
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto release = [&planner](uint32_t op_index, uint32_t operand_index) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.release(mem_idx);
  };

  // 0 CLAIM - 10
  claim(0, 0, 10, 0);

  // 1 CLAIM - 20
  claim(1, 0, 20, 10);

  // 2 CLAIM - 30
  claim(2, 2, 30, 30);

  // 0 RELEASE - 10
  release(0, 0);

  // 0 RELEASE again
  auto on_only_debug_mode = [&release]() {
    EXPECT_EXIT({ release(0, 0); }, ::testing::KilledBySignal(SIGABRT),
                "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };
  assert(on_only_debug_mode());
}

TEST(WICPlanner, claim_release_test)
{
  WICPlanner planner;

  auto claim = [&planner](uint32_t op_index, uint32_t operand_index, size_t size) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.claim(mem_idx, size);
  };

  auto release = [&planner](uint32_t op_index, uint32_t operand_index) {
    DisposableTensorIndex mem_idx{OperationIndex{op_index}, OperandIndex{operand_index}};
    planner.release(mem_idx);
  };

  auto verify = [&planner](uint32_t op_index, uint32_t operand_index, uint32_t size,
                           uint32_t expected_offset) {
    DisposableTensorIndex mem_idx(OperationIndex{op_index}, OperandIndex{operand_index});
    auto mem_blk = planner.memory_plans()[mem_idx];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  };

  auto capacity = [&planner](uint32_t expected_capacity) {
    auto actual_capacity = planner.capacity();
    ASSERT_EQ(actual_capacity, expected_capacity);
  };

  claim(0, 0, 20);
  claim(1, 0, 5);
  release(0, 0);
  claim(2, 2, 10);
  release(1, 0);
  claim(3, 1, 10);
  release(2, 2);
  claim(4, 1, 10);
  release(3, 1);
  claim(5, 1, 20);
  release(4, 1);
  claim(6, 1, 20);
  release(5, 1);

  // VERIFY 0 - 0
  verify(0, 0, 20, 0);

  // VERIFY 1 - 20
  verify(1, 0, 5, 20);

  // VERIFY 2 - 0
  verify(2, 2, 10, 0);

  // VERIFY 3 - 10
  verify(3, 1, 10, 10);

  // VERIFY 4 - 20
  verify(4, 1, 10, 20);

  // VERIFY 5 - 0
  verify(5, 1, 20, 0);

  // VERIFY 6 - 20
  verify(6, 1, 20, 20);

  // CAPACITY - 40
  capacity(40);
}
