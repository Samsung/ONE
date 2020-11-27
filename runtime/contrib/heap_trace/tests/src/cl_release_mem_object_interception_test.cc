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

extern std::unique_ptr<Trace> GlobalTrace;

namespace backstage
{

struct ClReleaseMemObjectStub : public TestEnv
{
  cl_context context;

  ClReleaseMemObjectStub() : TestEnv("./cl_release_mem_object_interception_test.log") {}

  void SetUp() final
  {
    cl_device_id device_id;
    int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    TestEnv::SetUp();
  }

  void TearDown() final
  {
    TestEnv::TearDown();

    clReleaseContext(context);
  }
};

TEST_F(ClReleaseMemObjectStub, should_work_as_standard_version)
{
  cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, NULL);
  clReleaseMemObject(mem);
  ASSERT_TRUE(mem);
}

TEST_F(ClReleaseMemObjectStub, must_log_deallocation_events_if_trace_is_ready_for_it)
{
  GlobalTrace.reset();
  cl_mem mem1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, NULL);
  ASSERT_TRUE(mem1);
  clReleaseMemObject(mem1);

  GlobalTrace.reset(new Trace);
  cl_mem mem2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128, NULL, NULL);
  cl_mem mem3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64, NULL, NULL);
  ASSERT_TRUE(mem2 && mem3);
  clReleaseMemObject(mem2);
  clReleaseMemObject(mem3);
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./cl_release_mem_object_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 192 B, Total allocated: 192 B, Total deallocated: 192 B\n");
}

TEST_F(ClReleaseMemObjectStub, must_log_deallocation_event_only_if_reference_counter_equals_to_zero)
{
  cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, NULL);
  clRetainMemObject(mem);
  clReleaseMemObject(mem);
  GlobalTrace.reset();
  ASSERT_STREQ(getContentOfFile("./cl_release_mem_object_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 1024 B, Total allocated: 1024 B, Total deallocated: 0 B\n");
  clReleaseMemObject(mem);

  GlobalTrace.reset(new Trace);
  mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, NULL);
  clRetainMemObject(mem);
  clReleaseMemObject(mem);
  clReleaseMemObject(mem);
  GlobalTrace.reset();
  ASSERT_STREQ(
    getContentOfFile("./cl_release_mem_object_interception_test.log").c_str(),
    "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
    "GPU - Peak mem usage: 1024 B, Total allocated: 1024 B, Total deallocated: 1024 B\n");
}

TEST_F(ClReleaseMemObjectStub, must_not_log_deallocation_event_if_original_function_failed)
{
  cl_mem mem;
  ASSERT_NE(clReleaseMemObject(mem), CL_SUCCESS);

  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./cl_release_mem_object_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
