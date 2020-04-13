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
#include "memory_pool_for_symbol_searcher_internals.h"

extern std::unique_ptr<Trace> GlobalTrace;

namespace backstage
{

struct FreeStub : public TestEnv
{
  FreeStub() : TestEnv("./free_interception_test.log") {}
};

TEST_F(FreeStub, should_work_as_standard_version)
{
  void *p = malloc(1024);
  free(p);
  ASSERT_TRUE(p);
  // TODO Bad test. Need use death test from Google test framework
}

TEST_F(FreeStub, must_log_deallocation_events_if_trace_is_ready_for_it)
{
  GlobalTrace.reset();
  void *p1 = malloc(1024);
  ASSERT_TRUE(p1);
  free(p1);

  GlobalTrace.reset(new Trace);
  void *p2 = malloc(128);
  void *p3 = malloc(64);
  ASSERT_TRUE(p2 && p3);
  free(p2);
  free(p3);
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./free_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 192 B, Total allocated: 192 B, Total deallocated: 192 "
               "B\nOn GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

TEST_F(FreeStub, can_deallocate_memory_using_pool_for_symbol_searcher_internals)
{

  MemoryPoolForSymbolSearcherInternals pool;
  void *volatile p1 = pool.allocate(1024);
  free(p1);
  void *volatile p2 = pool.allocate(1024);
  free(p2);
  GlobalTrace.reset();

  ASSERT_TRUE(p1 == p2);
  ASSERT_STREQ(getContentOfFile("./free_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 "
               "B\nOn GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
