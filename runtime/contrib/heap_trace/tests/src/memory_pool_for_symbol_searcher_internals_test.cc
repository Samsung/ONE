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

#include "gtest/gtest.h"

#include "memory_pool_for_symbol_searcher_internals.h"

namespace backstage
{

struct MemoryPoolForSymbolSearcherInternals : public ::testing::Test
{
};

TEST_F(MemoryPoolForSymbolSearcherInternals, can_help_users_allocate_deallocate_memory)
{
  ::MemoryPoolForSymbolSearcherInternals memory;

  void *p1 = memory.allocate(1024);

  ASSERT_TRUE(p1);
  memory.deallocate(p1);
}

TEST_F(MemoryPoolForSymbolSearcherInternals,
       should_reuse_memory_if_it_deallocated_just_after_allocations)
{
  ::MemoryPoolForSymbolSearcherInternals memory;

  void *p1 = memory.allocate(1024);
  memory.deallocate(p1);
  void *p2 = memory.allocate(128);
  memory.deallocate(p2);
  void *p3 = memory.allocate(3467);
  memory.deallocate(p3);

  ASSERT_TRUE(p1 && p2 && p3);
  ASSERT_TRUE(p1 == p2);
  ASSERT_TRUE(p2 == p3);
}

TEST_F(MemoryPoolForSymbolSearcherInternals,
       can_define_either_contains_memory_starting_from_incoming_pointer_or_not)
{
  ::MemoryPoolForSymbolSearcherInternals memory;

  void *p1 = memory.allocate(1024);
  void *p2 = malloc(1024);

  ASSERT_TRUE(memory.containsMemorySpaceStartedFromPointer(p1));
  ASSERT_FALSE(memory.containsMemorySpaceStartedFromPointer(p2));

  memory.deallocate(p1);
  free(p2);
}
}
