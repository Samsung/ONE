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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_MODEL_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_MODEL_H__

#include <string>

#include "absl/types/any.h"
#include "InternalTensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// There is yet another representation of CNN graph. The primary purpose of this
// representation is to simplify graph manipulation.

using ValueId = uint32_t;

// Used to emulate quantized behavior.
struct QuantizationParams
{
  float min = 0;
  float max = 0;
  float scale = 0;
};

struct Operation
{
  std::string type;
  absl::any attributes;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_MODEL_H__
