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

#include "DwConvolutionSelector.h"

#include "absl/memory/memory.h"
#include "open_cl/ClDevice.h"
#include "open_cl/kernels/DepthwiseConv.h"
#include "open_cl/kernels/DepthwiseConv3x3.h"
#include "open_cl/Precision.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{

std::unique_ptr<GPUOperation>
SelectDWConvolutionAdreno(const DepthwiseConvolution2DAttributes &attr,
                          const DeviceInfo &device_info, const OperationDef &op_def)
{
  if (IsDepthwiseConv3x3Supported(attr))
  {
    return absl::make_unique<DepthwiseConv3x3>(CreateDepthwiseConv3x3(device_info, op_def, attr));
  }
  else
  {
    return absl::make_unique<GPUOperation>(CreateDepthwiseConvolution2D(device_info, op_def, attr));
  }
}

std::unique_ptr<GPUOperation>
SelectDWConvolutionPowerVR(const DepthwiseConvolution2DAttributes &attr,
                           const DeviceInfo &device_info, const OperationDef &op_def)
{
  if (IsDepthwiseConv3x3Supported(attr))
  {
    return absl::make_unique<DepthwiseConv3x3>(CreateDepthwiseConv3x3(device_info, op_def, attr));
  }
  else
  {
    return absl::make_unique<GPUOperation>(CreateDepthwiseConvolution2D(device_info, op_def, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionMali(const DepthwiseConvolution2DAttributes &attr,
                                                      const DeviceInfo &device_info,
                                                      const OperationDef &op_def)
{
  const auto storage_type = op_def.src_tensors[0].storage_type;
  bool buffer_type =
    storage_type == TensorStorageType::BUFFER || storage_type == TensorStorageType::IMAGE_BUFFER;
  const MaliInfo mali_info = device_info.mali_info;
  if (IsDepthwiseConv3x3Supported(attr) && !mali_info.IsMidgard() && !buffer_type &&
      op_def.precision != CalculationsPrecision::F32)
  {
    return absl::make_unique<DepthwiseConv3x3>(CreateDepthwiseConv3x3(device_info, op_def, attr));
  }
  else
  {
    return absl::make_unique<GPUOperation>(CreateDepthwiseConvolution2D(device_info, op_def, attr));
  }
}
} // namespace

std::unique_ptr<GPUOperation> SelectDWConvolution(const DepthwiseConvolution2DAttributes &attr,
                                                  const DeviceInfo &device_info,
                                                  const OperationDef &op_def)
{
  if (device_info.IsAdreno())
  {
    return SelectDWConvolutionAdreno(attr, device_info, op_def);
  }
  else if (device_info.IsPowerVR())
  {
    return SelectDWConvolutionPowerVR(attr, device_info, op_def);
  }
  else if (device_info.IsMali())
  {
    return SelectDWConvolutionMali(attr, device_info, op_def);
  }
  else
  {
    return SelectDWConvolutionAdreno(attr, device_info, op_def);
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
