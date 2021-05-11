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

#ifndef __ONERT_BACKEND_GPU_CL_KERNELS_WROK_GROUP_PICKING_H__
#define __ONERT_BACKEND_GPU_CL_KERNELS_WROK_GROUP_PICKING_H__

#include <vector>

#include "TuningParameters.h"

#include "open_cl/ClKernel.h"
#include "open_cl/DeviceInfo.h"
#include "open_cl/Types.h"
#include "open_cl/WorkgroupSelection.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// multiplier can be power of two only
void GetPossibleWorkGroupsXYMultipleOf(int multiplier, const DeviceInfo &device_info,
                                       const KernelInfo &kernel_info, const int3 &grid,
                                       WorkGroupSizeAlignment z_alignment,
                                       std::vector<int3> *work_groups);

void GetPossibleWorkGroupsXMultipleOf(int multiplier, const DeviceInfo &device_info,
                                      const KernelInfo &kernel_info, const int3 &grid,
                                      WorkGroupSizeAlignment z_alignment,
                                      std::vector<int3> *work_groups);

int3 GetWorkGroupXY128ConvLinear(const int3 &grid);

int3 GetWorkGroupXY128Simple(const int3 &grid);
int3 GetWorkGroupXY128Conv(const int3 &grid);

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height);

void GetPossibleWorkGroups(TuningType tuning_type, const DeviceInfo &device_info,
                           const KernelInfo &kernel_info, const int3 &grid,
                           std::vector<int3> *work_groups);

void GetPossibleWorkGroupsConv(TuningType tuning_type, const DeviceInfo &device_info,
                               const KernelInfo &kernel_info, const int3 &grid,
                               std::vector<int3> *work_groups);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_KERNELS_WROK_GROUP_PICKING_H__
