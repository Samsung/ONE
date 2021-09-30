/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

using namespace tflite::gpu;

namespace onert
{
namespace backend
{
namespace gpu_cl
{

enum TensorType
{
  TENSOR_TYPE_VALID = 0,
  TENSOR_TYPE_INPUT = 1,
  TENSOR_TYPE_OUTPUT = 2,
  TENSOR_TYPE_DELETE = 3
};

absl::Status ExtractAxisFromIndex(int dims, int index, Axis *axis);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__
