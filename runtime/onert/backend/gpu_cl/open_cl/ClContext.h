/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_CL_CONTEXT_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_CL_CONTEXT_H__

#include "ClDevice.h"
#include "OpenclWrapper.h"
#include "DataType.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// A RAII wrapper around opencl context
class CLContext
{
public:
  CLContext() {}
  CLContext(cl_context context, bool has_ownership);

  // Move only
  CLContext(CLContext &&context);
  CLContext &operator=(CLContext &&context);
  CLContext(const CLContext &) = delete;
  CLContext &operator=(const CLContext &) = delete;

  ~CLContext();

  cl_context context() const { return context_; }

  bool IsFloatTexture2DSupported(int num_channels, DataType data_type,
                                 cl_mem_flags flags = CL_MEM_READ_WRITE) const;

private:
  void Release();

  cl_context context_ = nullptr;
  bool has_ownership_ = false;
};

absl::Status CreateCLContext(const CLDevice &device, CLContext *result);
absl::Status CreateCLGLContext(const CLDevice &device, cl_context_properties egl_context,
                               cl_context_properties egl_display, CLContext *result);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_CL_CONTEXT_H__
