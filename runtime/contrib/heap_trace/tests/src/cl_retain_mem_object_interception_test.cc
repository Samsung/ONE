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

#include <CL/cl.h>

extern std::unique_ptr<Trace> GlobalTrace;

namespace backstage
{

struct ClRetainMemObjectStub : public TestEnv
{
  cl_context context;

  ClRetainMemObjectStub() : TestEnv("cl_retain_mem_object_interception_test.log") {}

  void SetUp() final
  {
    cl_device_id device_id;
    int err = clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &err);
    TestEnv::SetUp();
  }

  void TearDown() final
  {
    TestEnv::TearDown();
    clReleaseContext(context);
  }
};

TEST_F(ClRetainMemObjectStub, must_work_as_standard_version)
{
  cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 1024, nullptr, nullptr);
  cl_int retain_mem_result = clRetainMemObject(mem);
  cl_int release_mem_result1 = clReleaseMemObject(mem);
  cl_int release_mem_result2 = clReleaseMemObject(mem);

  cl_mem bad_mem_object = nullptr;
  cl_int retain_mem_result_with_bad_mem_object = clRetainMemObject(bad_mem_object);

  ASSERT_TRUE(mem);
  ASSERT_TRUE(retain_mem_result == CL_SUCCESS);
  ASSERT_TRUE(release_mem_result1 == CL_SUCCESS);
  ASSERT_TRUE(release_mem_result2 == CL_SUCCESS);
  ASSERT_TRUE(retain_mem_result_with_bad_mem_object == CL_INVALID_MEM_OBJECT);
}

TEST_F(ClRetainMemObjectStub, must_do_not_log_new_allocation_event_just_increase_reference_count)
{
  GlobalTrace.reset();
  cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, nullptr, nullptr);

  GlobalTrace.reset(new Trace);
  clRetainMemObject(mem);
  GlobalTrace.reset();

  cl_int release_mem_result1 = clReleaseMemObject(mem);
  cl_int release_mem_result2 = clReleaseMemObject(mem);
  ASSERT_TRUE(release_mem_result1 == CL_SUCCESS);
  ASSERT_TRUE(release_mem_result2 == CL_SUCCESS);
  ASSERT_STREQ(getContentOfFile("cl_retain_mem_object_interception_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");
}

} // namespace backstage
