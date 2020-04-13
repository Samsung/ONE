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

struct ClCreateBufferStub : public TestEnv
{
  cl_context context;

  ClCreateBufferStub() : TestEnv("./cl_create_buffer_interception_test.log") {}

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

TEST_F(ClCreateBufferStub, must_allocate_space_as_standard_ocl_function)
{
  cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 1024, NULL, NULL);

  ASSERT_TRUE(mem);

  clReleaseMemObject(mem);
}

TEST_F(ClCreateBufferStub, must_log_allocation_events_if_trace_is_ready_for_it)
{
  GlobalTrace.reset();
  clReleaseMemObject(clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, NULL));

  GlobalTrace.reset(new Trace);
  clReleaseMemObject(clCreateBuffer(context, CL_MEM_READ_WRITE, 128, NULL, NULL));
  clReleaseMemObject(clCreateBuffer(context, CL_MEM_READ_WRITE, 64, NULL, NULL));
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./cl_create_buffer_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 128 B, Total allocated: 192 B, Total deallocated: 192 B\n");
}

TEST_F(ClCreateBufferStub,
       must_not_do_the_record_about_allocation_event_if_original_function_failed)
{
  cl_context badContext = nullptr;
  cl_mem p = clCreateBuffer(badContext, CL_MEM_READ_WRITE, 1024, nullptr, nullptr);
  GlobalTrace.reset();

  ASSERT_FALSE(p);
  ASSERT_STREQ(getContentOfFile("./cl_create_buffer_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
