/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DisposableTensorIndex.h"
#include "LayerScopeTensorIndex.h"
#include "MemoryPlanner.h"
#include "ir/Index.h"

using namespace onert::backend::train;
using onert::ir::OperandIndex;
using onert::ir::OperationIndex;

namespace
{

template <typename T> T to_index(uint32_t first, uint32_t second) = delete;

template <> DisposableTensorIndex to_index<DisposableTensorIndex>(uint32_t first, uint32_t second)
{
  return DisposableTensorIndex{OperationIndex{first}, OperandIndex{second}};
}

template <> LayerScopeTensorIndex to_index<LayerScopeTensorIndex>(uint32_t first, uint32_t second)
{
  return LayerScopeTensorIndex{OperationIndex{first}, second};
}

template <template <typename> class Planner, typename Index> class PlannerVerifier final
{
public:
  void claim(uint32_t first_idx, uint32_t second_idx, size_t size)
  {
    auto index = to_index<Index>(first_idx, second_idx);
    _planner.claim(index, size);
  }

  // Claim plan and verify newly added plan, call ASSERT_* on failure
  void claim(uint32_t first_idx, uint32_t second_idx, size_t size, uint32_t expected_offset)
  {
    auto index = to_index<Index>(first_idx, second_idx);
    _planner.claim(index, size);

    auto mem_blk = _planner.memory_plans()[index];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, size);
  }

  void release(uint32_t first_idx, uint32_t second_idx)
  {
    _planner.release(to_index<Index>(first_idx, second_idx));
  }

  // Verify capacity, call ASSERT_* on failure
  void capacity(uint32_t expected_capacity)
  {
    auto actual_capacity = _planner.capacity();
    ASSERT_EQ(actual_capacity, expected_capacity);
  }

  // Verify memory_plans's size and offset, calls ASSERT_* on failure
  void verify(uint32_t first_idx, uint32_t second_idx, size_t expected_size,
              uint32_t expected_offset)
  {
    auto index = to_index<Index>(first_idx, second_idx);
    auto mem_blk = _planner.memory_plans()[index];
    ASSERT_EQ(mem_blk.offset, expected_offset);
    ASSERT_EQ(mem_blk.size, expected_size);
  }

private:
  Planner<Index> _planner;
};

} // namespace

TEST(BumpPlanner, disposable_claim_test)
{
  PlannerVerifier<BumpPlanner, DisposableTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    p.claim(0, 0, 10, 0);
    p.claim(1, 0, 20, 10);
    p.claim(2, 2, 30, 30);
    p.release(0, 0);
    p.capacity(60);
  });
}

TEST(FirstFitPlanner, disposable_claim_release_test)
{
  PlannerVerifier<FirstFitPlanner, DisposableTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // 0 RELEASE - 10
    p.release(0, 0);

    // 3 CLAIM - 20
    p.claim(3, 1, 20, 60);

    // 4 CLAIM - 5
    p.claim(4, 1, 5, 0);

    // 5 CLAIM - 10
    p.claim(5, 1, 10, 80);

    // 6 CLAIM - 5
    p.claim(6, 1, 5, 5);

    // 2 RELEASE - 30
    p.release(2, 2);

    // 7 CLAIM - 35
    p.claim(7, 1, 35, 90);

    // 8 CLAIM - 10
    p.claim(8, 1, 10, 30);

    // 4 RELEASE - 5
    p.release(4, 1);

    // 9 CLAIM - 10
    p.claim(9, 0, 10, 40);

    // 10 CLAIM - 10
    p.claim(10, 0, 10, 50);

    // 6 RELEASE
    p.release(6, 1);

    // 1 RELEASE
    p.release(1, 0);

    // 8 RELEASE
    p.release(8, 1);

    // 9 RELEASE
    p.release(9, 0);

    // 10 RELEASE
    p.release(10, 0);

    // 3 RELEASE
    p.release(3, 1);

    // 5 RELEASE
    p.release(5, 1);

    // 7 RELEASE
    p.release(7, 1);

    // CAPACITY - 125
    p.capacity(125);
  });
}

TEST(FirstFitPlanner, neg_disposable_release_non_existing_index)
{
  PlannerVerifier<FirstFitPlanner, DisposableTensorIndex> p;

  auto on_only_debug_mode = [&p]() {
    EXPECT_DEATH({ p.release(0, 1); },
                 "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // RELEASE non-existing index
    assert(on_only_debug_mode());
  });
}

TEST(FirstFitPlanner, neg_disposable_release_twice)
{
  PlannerVerifier<FirstFitPlanner, DisposableTensorIndex> p;

  auto on_only_debug_mode = [&p]() {
    EXPECT_EXIT({ p.release(0, 0); }, ::testing::KilledBySignal(SIGABRT),
                "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // 0 RELEASE - 10
    p.release(0, 0);

    // 0 RELEASE again
    assert(on_only_debug_mode());
  });
}

TEST(WICPlanner, disposable_claim_release_test)
{
  PlannerVerifier<WICPlanner, DisposableTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    p.claim(0, 0, 20);
    p.claim(1, 0, 5);
    p.release(0, 0);
    p.claim(2, 2, 10);
    p.release(1, 0);
    p.claim(3, 1, 10);
    p.release(2, 2);
    p.claim(4, 1, 10);
    p.release(3, 1);
    p.claim(5, 1, 20);
    p.release(4, 1);
    p.claim(6, 1, 20);
    p.release(5, 1);

    // VERIFY 0 - 0
    p.verify(0, 0, 20, 0);

    // VERIFY 1 - 20
    p.verify(1, 0, 5, 20);

    // VERIFY 2 - 0
    p.verify(2, 2, 10, 0);

    // VERIFY 3 - 10
    p.verify(3, 1, 10, 10);

    // VERIFY 4 - 20
    p.verify(4, 1, 10, 20);

    // VERIFY 5 - 0
    p.verify(5, 1, 20, 0);

    // VERIFY 6 - 20
    p.verify(6, 1, 20, 20);

    // CAPACITY - 40
    p.capacity(40);
  });
}

TEST(BumpPlanner, layerscope_claim_test)
{
  PlannerVerifier<BumpPlanner, LayerScopeTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    p.claim(0, 0, 10, 0);
    p.claim(1, 0, 20, 10);
    p.claim(2, 2, 30, 30);
    p.release(0, 0);
    p.capacity(60);
  });
}

TEST(FirstFitPlanner, layerscope_claim_release_test)
{
  PlannerVerifier<FirstFitPlanner, LayerScopeTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // 0 RELEASE - 10
    p.release(0, 0);

    // 3 CLAIM - 20
    p.claim(3, 1, 20, 60);

    // 4 CLAIM - 5
    p.claim(4, 1, 5, 0);

    // 5 CLAIM - 10
    p.claim(5, 1, 10, 80);

    // 6 CLAIM - 5
    p.claim(6, 1, 5, 5);

    // 2 RELEASE - 30
    p.release(2, 2);

    // 7 CLAIM - 35
    p.claim(7, 1, 35, 90);

    // 8 CLAIM - 10
    p.claim(8, 1, 10, 30);

    // 4 RELEASE - 5
    p.release(4, 1);

    // 9 CLAIM - 10
    p.claim(9, 0, 10, 40);

    // 10 CLAIM - 10
    p.claim(10, 0, 10, 50);

    // 6 RELEASE
    p.release(6, 1);

    // 1 RELEASE
    p.release(1, 0);

    // 8 RELEASE
    p.release(8, 1);

    // 9 RELEASE
    p.release(9, 0);

    // 10 RELEASE
    p.release(10, 0);

    // 3 RELEASE
    p.release(3, 1);

    // 5 RELEASE
    p.release(5, 1);

    // 7 RELEASE
    p.release(7, 1);

    // CAPACITY - 125
    p.capacity(125);
  });
}

TEST(FirstFitPlanner, neg_layerscope_release_non_existing_index)
{
  PlannerVerifier<FirstFitPlanner, LayerScopeTensorIndex> p;

  auto on_only_debug_mode = [&p]() {
    EXPECT_DEATH({ p.release(0, 1); },
                 "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // RELEASE non-existing index
    assert(on_only_debug_mode());
  });
}

TEST(FirstFitPlanner, neg_layerscope_release_twice)
{
  PlannerVerifier<FirstFitPlanner, LayerScopeTensorIndex> p;

  auto on_only_debug_mode = [&p]() {
    EXPECT_EXIT({ p.release(0, 0); }, ::testing::KilledBySignal(SIGABRT),
                "Cannot release for given index. It has been not claimed or released already.");
    return true;
  };

  ASSERT_NO_FATAL_FAILURE({
    // 0 CLAIM - 10
    p.claim(0, 0, 10, 0);

    // 1 CLAIM - 20
    p.claim(1, 0, 20, 10);

    // 2 CLAIM - 30
    p.claim(2, 2, 30, 30);

    // 0 RELEASE - 10
    p.release(0, 0);

    // 0 RELEASE again
    assert(on_only_debug_mode());
  });
}

TEST(WICPlanner, layerscope_claim_release_test)
{
  PlannerVerifier<WICPlanner, LayerScopeTensorIndex> p;

  ASSERT_NO_FATAL_FAILURE({
    p.claim(0, 0, 20);
    p.claim(1, 0, 5);
    p.release(0, 0);
    p.claim(2, 2, 10);
    p.release(1, 0);
    p.claim(3, 1, 10);
    p.release(2, 2);
    p.claim(4, 1, 10);
    p.release(3, 1);
    p.claim(5, 1, 20);
    p.release(4, 1);
    p.claim(6, 1, 20);
    p.release(5, 1);

    // VERIFY 0 - 0
    p.verify(0, 0, 20, 0);

    // VERIFY 1 - 20
    p.verify(1, 0, 5, 20);

    // VERIFY 2 - 0
    p.verify(2, 2, 10, 0);

    // VERIFY 3 - 10
    p.verify(3, 1, 10, 10);

    // VERIFY 4 - 20
    p.verify(4, 1, 10, 20);

    // VERIFY 5 - 0
    p.verify(5, 1, 20, 0);

    // VERIFY 6 - 20
    p.verify(6, 1, 20, 20);

    // CAPACITY - 40
    p.capacity(40);
  });
}
