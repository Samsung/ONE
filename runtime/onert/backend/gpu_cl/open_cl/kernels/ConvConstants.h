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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_CONSTANTS_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_CONSTANTS_H__

#include "open_cl/Buffer.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/LinearStorage.h"
#include "open_cl/Tensor.h"
#include "open_cl/Util.h"
#include "open_cl/DataType.h"
#include "open_cl/Operations.h"
#include "open_cl/Shape.h"
#include "open_cl/Status.h"
#include "open_cl/Tensor.h"
#include "open_cl/Types.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <DataType S, typename T>
void RearrangeWeightsForConvConstants(const InternalTensor<OHWI, S> &weights, absl::Span<T> dst)
{
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s)
  {
    for (int y = 0; y < kernel_y; ++y)
    {
      for (int x = 0; x < kernel_x; ++x)
      {
        for (int d = 0; d < dst_depth; ++d)
        {
          const int channels_count = std::min(4, weights.shape.i - s * 4);
          T filters[4];
          for (int i = 0; i < 4; ++i)
          {
            for (int j = 0; j < channels_count; ++j)
            {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o)
              {
                const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[i][j] = weights.data[f_index];
              }
              else
              {
                filters[i][j] = 0.0f;
              }
            }
          }
          T filters_new[4];
          for (int i = 0; i < 4; ++i)
          {
            for (int j = 0; j < 4; ++j)
            {
              filters_new[i][j] = filters[j][i];
            }
          }
          for (int i = 0; i < channels_count; ++i)
          {
            dst[counter++] = filters_new[i];
          }
        }
      }
    }
  }
}

template <DataType T>
void UploadWeightsForConvConstants(const InternalTensor<OHWI, T> &weights,
                                   CalculationsPrecision precision, GPUOperation *op)
{
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const bool f32_weights = precision == CalculationsPrecision::F32;
  const int float_size = f32_weights ? 4 : 2;
  const int float_count = weights.shape.i * dst_depth * 4 * kernel_x * kernel_y;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = float_size * float_count;
  desc.data.resize(desc.size);

  if (f32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(desc.data.data());
    RearrangeWeightsForConvConstants(weights, absl::MakeSpan(ptr, float_count / 4));
  }
  //   else
  //   {
  //     half4 *ptr = reinterpret_cast<half4 *>(desc.data.data());
  //     RearrangeWeightsForConvConstants(weights, absl::MakeSpan(ptr, float_count / 4));
  //   }

  op->args_.AddObject("weigths", absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvConstantsSupported(const DeviceInfo &device_info, const OperationDef &definition,
                              const Convolution2DAttributes &attr);

GPUOperation CreateConvConstants(const DeviceInfo &device_info, const OperationDef &definition,
                                 const Convolution2DAttributes &attr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_CONSTANTS_H__
