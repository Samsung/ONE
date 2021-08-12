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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_POOLING_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_POOLING_H__

#include "GpuOperation.h"

#include "open_cl/Operations.h"
#include "open_cl/Precision.h"
#include "open_cl/ClKernel.h"
#include "open_cl/Tensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

GPUOperation CreatePooling(const OperationDef &definition, const Pooling2DAttributes &attr);

GPUOperation CreatePooling(const OperationDef &definition, const Pooling3DAttributes &attr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_ADD_H__
