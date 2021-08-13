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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_SIMPLE_SELECTORS_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_SIMPLE_SELECTORS_H__

#include <memory>

#include "open_cl/ClDevice.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/Operations.h"
#include "open_cl/Shape.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

void SelectAdd(const OperationDef &op_def, const std::vector<int> &channels, int dst_channels,
               std::unique_ptr<GPUOperation> *ptr);

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes &attr, const OperationDef &op_def);

std::unique_ptr<GPUOperation>
SelectDWConvolutionDynamicWeights(const DepthwiseConvolution2DAttributes &attr,
                                  const DeviceInfo &device_info, const OperationDef &op_def);

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes &attr,
                                            const OperationDef &op_def);
void SelectSoftmax(const BHWC &shape, const OperationDef &op_def,
                   std::unique_ptr<GPUOperation> *ptr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_SIMPLE_SELECTORS_H__
