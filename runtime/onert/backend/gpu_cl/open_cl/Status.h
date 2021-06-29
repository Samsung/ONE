/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_STATUS_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_STATUS_H__

#include "absl/status/status.h" // IWYU pragma: export
#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok())           \
      return c;            \
  } // IWYU pragma: export

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_STATUS_H__
