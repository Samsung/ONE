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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_CONVOLUTION_SELECTOR_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_CONVOLUTION_SELECTOR_H__

#include <memory>

#include "open_cl/kernels/ConvCommon.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/ModelHints.h"
#include "open_cl/Operations.h"
#include "open_cl/Shape.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

std::unique_ptr<GPUOperation> SelectConvolution(const Convolution2DAttributes &attr,
                                                const BHWC &dst_shape,
                                                const DeviceInfo &device_info,
                                                const OperationDef &op_def, ModelHints hints);

std::unique_ptr<GPUOperation> SelectConvolutionForWinograd(const Convolution2DAttributes &attr,
                                                           const BHWC &dst_shape,
                                                           const DeviceInfo &device_info,
                                                           const OperationDef &op_def,
                                                           ModelHints hints);

std::unique_ptr<GPUOperation>
SelectConvolutionWithDynamicWeights(const Convolution2DAttributes &attr, const BHWC &weights_shape,
                                    const BHWC &dst_shape, const DeviceInfo &device_info,
                                    const OperationDef &op_def, ModelHints hints,
                                    ConvWeightsDescription *weights_desc);

std::unique_ptr<GPUOperation>
SelectConverterToConvWeights(const ConvWeightsDescription &weights_desc, const OperationDef &op_def,
                             ModelHints hints);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_SELECTORS_CONVOLUTION_SELECTOR_H__
