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

#include "common_test_environment.h"
#include "file_content_manipulations.h"

#include "trace.h"
#include "symbol_searcher.h"
#include "memory_pool_for_symbol_searcher_internals.h"

#include <limits>

extern std::unique_ptr<Trace> GlobalTrace;

namespace backstage
{

struct ReallocStub : public TestEnv
{
  ReallocStub() : TestEnv("./realloc_interception_test.log") {}
};

TEST_F(ReallocStub, must_allocate_space_as_standard_realloc)
{
  void *p = malloc(128);
  p = realloc(p, 1024);

  ASSERT_TRUE(p);
  free(p);
}

TEST_F(ReallocStub, must_log_allocation_deallocation_events_if_trace_is_ready_for_it)
{
  std::array<char, 1024> reference_data;
  reference_data.fill('a');
  void *p1 = malloc(1024);
  memcpy(p1, reference_data.data(), reference_data.size());
  void *p2 = realloc(p1, 64);
  void *p3 = realloc(p2, 128);
  GlobalTrace.reset();

  ASSERT_TRUE(p3);
  ASSERT_TRUE(memcmp(p3, reference_data.data(), 64) == 0);
  ASSERT_STREQ(getContentOfFile("./realloc_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 1024 B, Total allocated: 1216 B, Total deallocated: 1088 "
               "B\nOn GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
  free(p3);
}

TEST_F(ReallocStub,
       must_not_do_the_record_about_allocation_deallocation_events_if_original_function_failed)
{
  GlobalTrace.reset();
  void *p = malloc(128);
  GlobalTrace.reset(new Trace);

  void *ptr_after_realloc = realloc(p, std::numeric_limits<size_t>::max());
  ptr_after_realloc = realloc(p, 0);
  GlobalTrace.reset();

  ASSERT_FALSE(ptr_after_realloc);
  ASSERT_STREQ(getContentOfFile("./realloc_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");

  free(p);
}

TEST_F(ReallocStub, should_work_as_malloc_when_incoming_ptr_is_equal_to_nullptr)
{
  void *p = realloc(nullptr, 1024);
  GlobalTrace.reset();

  ASSERT_TRUE(p);
  ASSERT_STREQ(
    getContentOfFile("./realloc_interception_test.log").c_str(),
    "On CPU - Peak heap usage: 1024 B, Total allocated: 1024 B, Total deallocated: 0 B\nOn "
    "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");

  free(p);
}

TEST_F(
  ReallocStub,
  should_not_influence_on_trace_results_even_if_orignal_function_return_any_not_null_ptr_when_incoming_size_is_zero_and_ptr_is_null)
{
  void *p = realloc(nullptr, 0);
  free(p);
  GlobalTrace.reset();

  ASSERT_TRUE(p);
  ASSERT_STREQ(getContentOfFile("./realloc_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

TEST_F(ReallocStub, should_allocate_memory_from_pool_for_symbol_searcher_internal_usage_if_need)
{
  signalizeThatNextAllocationsWillBeForSymbolSearcherInternalUsage();
  void *p = malloc(128);
  p = realloc(p, 1024);
  signalizeThatSymbolSearcherEndedOfWork();
  GlobalTrace.reset();

  MemoryPoolForSymbolSearcherInternals pool;
  ASSERT_TRUE(p);
  ASSERT_TRUE(pool.containsMemorySpaceStartedFromPointer(p));
  ASSERT_STREQ(getContentOfFile("./realloc_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
