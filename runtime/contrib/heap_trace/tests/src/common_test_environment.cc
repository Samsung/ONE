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

#include "trace.h"

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

extern std::unique_ptr<Trace> GlobalTrace;

void TestEnv::SetUp() { configureTraceToMakeLogInFile(); }

void TestEnv::TearDown() { removeOldTraceLogIfNeed(); }

void TestEnv::configureTraceToMakeLogInFile()
{
  removeOldTraceLogIfNeed();
  setNewNameOfTraceLog();
}

void TestEnv::removeOldTraceLogIfNeed()
{
  GlobalTrace.reset();
  const char *trace_log_name = getenv("HEAP_TRACE_LOG");
  if (trace_log_name)
  {
    fs::remove(trace_log_name);
  }
}

void TestEnv::setNewNameOfTraceLog()
{
  setenv("HEAP_TRACE_LOG", test_log_file.c_str(), 1);
  GlobalTrace.reset(new ::Trace);
}
