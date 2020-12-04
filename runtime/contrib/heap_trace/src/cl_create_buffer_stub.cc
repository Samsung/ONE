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

#include "trace.h"
#include "function_resolver.h"

#include <CL/cl.h>

#include <memory>

extern std::unique_ptr<Trace> GlobalTrace;

extern "C" {

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
                      cl_int *errcode_ret)
{
  static auto isOriginalFunctionCallSuccessful = [](cl_mem result) -> bool { return result; };

  static auto originalFunction =
    findFunctionByName<cl_mem, cl_context, cl_mem_flags, size_t, void *, cl_int *>(
      "clCreateBuffer");
  cl_mem result = originalFunction(context, flags, size, host_ptr, errcode_ret);
  if (isOriginalFunctionCallSuccessful(result) && !Trace::Guard{}.isActive())
  {
    GlobalTrace->logAllocationEvent(result, size);
  }

  return result;
}
}
