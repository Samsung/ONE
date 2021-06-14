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

#ifndef __ONERT_BACKEND_GPU_CL_KERNELS_UTIL_H__
#define __ONERT_BACKEND_GPU_CL_KERNELS_UTIL_H__

#include <string>
#include <vector>

#include "open_cl/DeviceInfo.h"
#include "open_cl/Precision.h"
#include "open_cl/DataType.h"
#include "open_cl/Shape.h"
#include "open_cl/Tensor.h"
#include "open_cl/Types.h"
#include "open_cl/Util.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

std::string GetCommonDefines(CalculationsPrecision precision);

// Calculates correct X coordinate when stride != 1 and batch != 1 for layouts
// with B after W (for example HWBC4) and WB stored in one axis of GPU
// resources.
std::string GetXStrideCorrected(const std::string &src_x, const std::string &batch_size,
                                const std::string &stride_x, const std::string &padding_x);

// Calculates correct X coordinate when stride != 1 and batch != 1 for layouts
// with B after W (for example HWBC4) and WB stored in one axis of GPU
// resources.
std::string GetXStrideCorrectedV2(const std::string &src_x, const std::string &batch_size,
                                  const std::string &stride_x, const std::string &padding_x);

// Returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);

// returns first work group from wgs that has size not bigger than max_wg_size
// if no suitable groups among wgs, returns {1, 1, 1}
int3 GetFirstSuitableWorkGroup(const std::vector<int3> &wgs, int max_wg_size);

// task_size as amount of FLT4 processed elements.
int GetRecommendedBlockSizeForConv(const DeviceInfo &device, CalculationsPrecision precision,
                                   int task_size);

int3 GetWorkGroupsCount(const int3 &grid_size, const int3 &work_group_size);
} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_KERNELS_UTIL_H__
