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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_

#include "DeviceInfo.h"
#include "TensorType.h"
#include "DataType.h"
#include "Shape.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

bool CanCreateTensorWithShape(const DeviceInfo &device_info, const BHWDC &shape,
                              const TensorDescriptor &descriptor);

bool CanCreateTensorWithShape(const DeviceInfo &device_info, const BHWC &shape,
                              const TensorDescriptor &descriptor);

TensorStorageType SelectBestStorageType(const DeviceInfo &device_info, const BHWC &shape,
                                        const TensorStorageType &desired, const DataType &data_type,
                                        const Layout &layout);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_
