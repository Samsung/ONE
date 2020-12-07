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

#include <CL/cl.h>

#include <experimental/filesystem>
#include <thread>
#include <atomic>

using namespace std;
namespace fs = experimental::filesystem;

extern unique_ptr<::Trace> GlobalTrace;

namespace backstage
{

struct Trace : TestEnv
{
  Trace() : TestEnv("./trace_test.log") {}

  void generateGarbageInTestLogFile();
  template <typename MemType>
  static void emulateAllocationEvent(size_t eventsPoolId, size_t numberOfEmulation,
                                     size_t numberOfBytesPerOneEmulation, atomic_bool &isPauseNeed);
};

TEST_F(Trace,
       must_create_log_file_with_name_defined_in_env_var_HEAP_TRACE_LOG_during_initialization)
{
  ASSERT_TRUE(fs::exists("./trace_test.log"));
}

TEST_F(Trace, must_truncate_log_file_if_it_exists_during_initialization)
{
  GlobalTrace.reset();
  generateGarbageInTestLogFile();
  GlobalTrace.reset(new ::Trace);
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./trace_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

void Trace::generateGarbageInTestLogFile()
{
  ofstream log("./trace_test.log");
  log << string(256, 'a');
}

TEST_F(Trace, should_not_generate_any_records_in_log_during_creation)
{
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./trace_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

TEST_F(Trace, can_signalize_to_users_if_it_is_ready_for_using)
{
  ASSERT_FALSE(::Trace::Guard().isActive());
}

TEST_F(Trace, must_signalize_that_it_is_not_ready_for_using_until_it_is_not_created)
{
  GlobalTrace.reset();
  ASSERT_TRUE(::Trace::Guard().isActive());
}

TEST_F(Trace, should_work_correctly_in_multithreaded_environment)
{
  constexpr size_t numberOfThreads = 10, numberOfEmulations = 100,
                   numberOfBytesPerOneEmulation = 1024;
  atomic_bool isPauseNeed{true};
  array<thread, numberOfThreads> threads;
  for (size_t i = 0; i < numberOfThreads / 2; ++i)
  {
    threads[i] = thread(emulateAllocationEvent<void *>, i, numberOfEmulations,
                        numberOfBytesPerOneEmulation, ref(isPauseNeed));
  }
  for (size_t i = numberOfThreads / 2; i < numberOfThreads; ++i)
  {
    threads[i] = thread(emulateAllocationEvent<cl_mem>, i, numberOfEmulations,
                        numberOfBytesPerOneEmulation, ref(isPauseNeed));
  }

  GlobalTrace.reset(new ::Trace);
  isPauseNeed = false;

  for (size_t i = 0; i < numberOfThreads; ++i)
  {
    threads[i].join();
  }
  GlobalTrace.reset();

  string thisShouldBeInLogFile =
    "Total allocated: " +
    to_string(numberOfThreads / 2 * numberOfEmulations * numberOfBytesPerOneEmulation) +
    " B, Total deallocated: " +
    to_string(numberOfThreads / 2 * numberOfEmulations * numberOfBytesPerOneEmulation) + " B\n";
  string andThisToo =
    "Total allocated: " +
    to_string(numberOfThreads / 2 * numberOfEmulations * numberOfBytesPerOneEmulation) +
    " B, Total deallocated: " +
    to_string(numberOfThreads / 2 * numberOfEmulations * numberOfBytesPerOneEmulation) + " B\n";
  ASSERT_TRUE(getContentOfFile("./trace_test.log").find(thisShouldBeInLogFile) != string::npos);
  ASSERT_TRUE(getContentOfFile("./trace_test.log").find(andThisToo) != string::npos);
}

template <typename MemType>
void Trace::emulateAllocationEvent(size_t eventsPoolId, size_t numberOfEmulation,
                                   size_t numberOfBytesPerOneEmulation, atomic_bool &isPauseNeed)
{
  while (isPauseNeed)
  {
    continue;
  }

  for (size_t i = 1; i <= numberOfEmulation; ++i)
  {
    GlobalTrace->logAllocationEvent((MemType)(i + numberOfEmulation * eventsPoolId),
                                    numberOfBytesPerOneEmulation);
  }

  for (size_t i = 1; i <= numberOfEmulation; ++i)
  {
    GlobalTrace->logDeallocationEvent((MemType)(i + numberOfEmulation * eventsPoolId));
  }
}

TEST_F(Trace, must_log_allocation_and_deallocation_events)
{
  void *memOnCPU1 = (void *)1, *memOnCPU2 = (void *)3;
  cl_mem memOnGPU1 = (cl_mem)2, memOnGPU2 = (cl_mem)4;
  GlobalTrace->logAllocationEvent(memOnCPU1, 347);
  GlobalTrace->logDeallocationEvent(memOnCPU1);
  GlobalTrace->logAllocationEvent(memOnGPU2, 592);
  GlobalTrace->logDeallocationEvent(memOnGPU2);
  GlobalTrace->logAllocationEvent(memOnGPU1, 349);
  GlobalTrace->logDeallocationEvent(memOnGPU1);
  GlobalTrace->logAllocationEvent(memOnCPU2, 568);
  GlobalTrace->logDeallocationEvent(memOnCPU2);
  GlobalTrace.reset();

  string shouldBeInLogFile = "On CPU - Peak heap usage: " + to_string(568) +
                             " B, Total allocated: " + to_string(347 + 568) +
                             " B, Total deallocated: " + to_string(347 + 568) +
                             " B\n"
                             "On GPU - Peak mem usage: " +
                             to_string(592) + " B, Total allocated: " + to_string(592 + 349) +
                             " B, Total deallocated: " + to_string(592 + 349) + " B\n";
  ASSERT_STREQ(getContentOfFile("./trace_test.log").c_str(), shouldBeInLogFile.c_str());
}

} // namespace backstage
