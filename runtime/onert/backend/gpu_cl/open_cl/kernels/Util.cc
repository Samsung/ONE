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

#include "Util.h"

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include "open_cl/Precision.h"
#include "open_cl/DataType.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

std::string GetCommonDefines(CalculationsPrecision precision)
{
  std::string result;

  switch (precision)
  {
    case CalculationsPrecision::F32:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define FLT float\n";
      result += "#define FLT2 float2\n";
      result += "#define FLT3 float3\n";
      result += "#define FLT4 float4\n";
      result += "#define TO_FLT4 convert_float4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      break;
    case CalculationsPrecision::F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 half4\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_half4\n";
      result += "#define TO_ACCUM_FLT convert_half\n";
      break;
    case CalculationsPrecision::F32_F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      break;
  }
  return result;
}

float4 GetMaskForLastPlane(int channels)
{
  float4 mask = float4(0.0f);
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i)
  {
    mask[i] = 1.0f;
  }
  return mask;
}

int3 GetFirstSuitableWorkGroup(const std::vector<int3> &wgs, int max_wg_size)
{
  for (const auto &wg : wgs)
  {
    const int wg_size = wg.x * wg.y * wg.z;
    if (wg_size <= max_wg_size)
    {
      return wg;
    }
  }
  return {1, 1, 1};
}

int GetRecommendedBlockSizeForConv(const DeviceInfo &device_info, CalculationsPrecision precision,
                                   int task_size)
{
  const float task_size_per_cu = task_size / static_cast<float>(device_info.compute_units_count);
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!device_info.IsMali())
  {
    return 1;
  }
  MaliInfo mali_info = device_info.mali_info;
  switch (precision)
  {
    case CalculationsPrecision::F16:
      if (mali_info.IsBifrostGen1())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      }
      else if (mali_info.IsBifrostGen2())
      {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      }
      else if (mali_info.IsBifrostGen3() || mali_info.IsValhall())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      }
      else if (mali_info.IsMidgard())
      {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsBifrostGen1())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      }
      else if (mali_info.IsBifrostGen2())
      {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      }
      else if (mali_info.IsBifrostGen3() || mali_info.IsValhall())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      }
      else if (mali_info.IsMidgard())
      {
        threshold_1 = 256.0f * 4.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsBifrostGen1())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      }
      else if (mali_info.IsBifrostGen2())
      {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      }
      else if (mali_info.IsBifrostGen3() || mali_info.IsValhall())
      {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      }
      else if (mali_info.IsMidgard())
      {
        threshold_1 = 256.0f * 16.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1)
  {
    block_size = 1;
  }
  else if (task_size_per_cu <= threshold_2)
  {
    block_size = 2;
  }
  else if (task_size_per_cu <= threshold_4)
  {
    block_size = 4;
  }
  else
  {
    block_size = 8;
  }
  return block_size;
}

int3 GetWorkGroupsCount(const int3 &grid_size, const int3 &work_group_size)
{
  int3 work_groups_count;
  work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
  work_groups_count.y = DivideRoundUp(grid_size.y, work_group_size.y);
  work_groups_count.z = DivideRoundUp(grid_size.z, work_group_size.z);
  return work_groups_count;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
