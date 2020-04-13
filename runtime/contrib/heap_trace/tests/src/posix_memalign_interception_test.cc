
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

#include "common_test_environment.h"
#include "file_content_manipulations.h"

#include "trace.h"
#include "symbol_searcher.h"
#include "memory_pool_for_symbol_searcher_internals.h"

#include <limits>

extern std::unique_ptr<Trace> GlobalTrace;

namespace backstage
{

struct PosixMemalignStub : public TestEnv
{
  PosixMemalignStub() : TestEnv("posix_memalign_interception_test.log") {}
};

TEST_F(PosixMemalignStub, must_allocate_space_as_standard_posix_memalign)
{
  void *p = nullptr;
  int res = posix_memalign(&p, 4, 12);

  ASSERT_TRUE(res == 0);
  ASSERT_TRUE(p);
  free(p);
}

TEST_F(PosixMemalignStub, must_log_allocation_events_if_trace_is_ready_for_it)
{
  GlobalTrace.reset();
  void *p1 = nullptr;
  int res1 = posix_memalign(&p1, 4, 1024);

  GlobalTrace.reset(new Trace);
  void *p2 = nullptr;
  int res2 = posix_memalign(&p2, 4, 128);
  void *p3 = nullptr;
  int res3 = posix_memalign(&p3, 4, 64);
  GlobalTrace.reset();

  ASSERT_TRUE(res1 == 0 && res2 == 0 && res3 == 0);
  ASSERT_TRUE(p1 && p2 && p3);
  ASSERT_STREQ(getContentOfFile("posix_memalign_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 192 B, Total allocated: 192 B, Total deallocated: 0 "
               "B\nOn GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
  free(p1);
  free(p2);
  free(p3);
}

TEST_F(PosixMemalignStub, must_not_do_the_record_about_allocation_event_if_original_function_failed)
{
  void *p = nullptr;
  int res = posix_memalign(&p, 1, std::numeric_limits<size_t>::max());
  GlobalTrace.reset();

  ASSERT_FALSE(res == 0);
  ASSERT_FALSE(p);
  ASSERT_STREQ(getContentOfFile("posix_memalign_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

TEST_F(PosixMemalignStub,
       should_allocate_memory_from_pool_for_symbol_searcher_internal_usage_if_need)
{
  signalizeThatNextAllocationsWillBeForSymbolSearcherInternalUsage();
  void *p = nullptr;
  int res = posix_memalign(&p, 1, 1024);
  signalizeThatSymbolSearcherEndedOfWork();
  GlobalTrace.reset();

  MemoryPoolForSymbolSearcherInternals pool;
  ASSERT_TRUE(res == 0);
  ASSERT_TRUE(p);
  ASSERT_TRUE(pool.containsMemorySpaceStartedFromPointer(p));
  ASSERT_STREQ(getContentOfFile("posix_memalign_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
