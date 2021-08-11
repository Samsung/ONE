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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_H__

#include <vector>

#include "open_cl/Buffer.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/LinearStorage.h"
#include "open_cl/Tensor.h"
#include "open_cl/Texture2d.h"
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
void RearrangeWeightsForDWConv2D(const InternalTensor<OHWI, S> &weights, absl::Span<T> dst)
{
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_depth = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d)
  {
    for (int y = 0; y < kernel_y; ++y)
    {
      for (int x = 0; x < kernel_x; ++x)
      {
        T filter_val;
        for (int i = 0; i < 4; ++i)
        {
          const int d_ch = d * 4 + i;
          if (d_ch < dst_channels)
          {
            const int f_index =
              weights.shape.LinearIndex({d_ch % weights.shape.o, y, x, d_ch / weights.shape.o});
            filter_val[i] = weights.data[f_index];
          }
          else
          {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }
  }
}

template <DataType T>
void UploadWeightsForDWConv2D(const InternalTensor<OHWI, T> &weights, bool weights_are_buffer,
                              CalculationsPrecision precision, GPUOperation *op)
{
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int elements_count = kernel_x * kernel_y * dst_slices;

  const bool fp32_weights = precision == CalculationsPrecision::F32;
  const int float4_size = fp32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);

  if (fp32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(data.data());
    RearrangeWeightsForDWConv2D(weights, absl::MakeSpan(ptr, elements_count));
  }
  // TODO
  // It doesn't support F16 yet. I will try to add it later.
  //
  // else {
  //   half4* ptr = reinterpret_cast<half4*>(data.data());
  //   RearrangeWeightsForDWConv2D(weights, absl::MakeSpan(ptr, elements_count));
  // }

  if (weights_are_buffer)
  {
    BufferDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    op->args_.AddObject("weights", absl::make_unique<BufferDescriptor>(desc));
  }
  else
  {
    Texture2DDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.size = int2(kernel_x * kernel_y, dst_slices);
    desc.data = std::move(data);
    op->args_.AddObject("weights", absl::make_unique<Texture2DDescriptor>(desc));
  }
}

template <DataType S, typename T>
void RearrangeWeightsForDWConv3D(const InternalTensor<OHWDI, S> &weights, absl::Span<T> dst)
{
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  const int kernel_z = weights.shape.d;

  int counter = 0;
  for (int d = 0; d < dst_slices; ++d)
  {
    for (int z = 0; z < kernel_z; ++z)
    {
      for (int y = 0; y < kernel_y; ++y)
      {
        for (int x = 0; x < kernel_x; ++x)
        {
          T filter_val;
          for (int i = 0; i < 4; ++i)
          {
            const int d_ch = d * 4 + i;
            if (d_ch < dst_channels)
            {
              const int f_index = weights.shape.LinearIndex(
                {d_ch % weights.shape.o, y, x, z, d_ch / weights.shape.o});
              filter_val[i] = weights.data[f_index];
            }
            else
            {
              filter_val[i] = 0.0f;
            }
          }
          dst[counter++] = filter_val;
        }
      }
    }
  }
}

template <DataType T>
void UploadWeightsForDWConv3D(const InternalTensor<OHWDI, T> &weights, bool weights_are_buffer,
                              CalculationsPrecision precision, GPUOperation *op)
{
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  const int kernel_z = weights.shape.d;

  const int elements_count = kernel_x * kernel_y * kernel_z * dst_slices;

  const bool fp32_weights = precision == CalculationsPrecision::F32;
  const int float4_size = fp32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);

  if (fp32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(data.data());
    RearrangeWeightsForDWConv3D(weights, absl::MakeSpan(ptr, elements_count));
  }
  // TODO
  // It doesn't support F16 yet. I will try to add it later.
  //
  // else {
  //   half4* ptr = reinterpret_cast<half4*>(data.data());
  //   RearrangeWeightsForDWConv3D(weights, absl::MakeSpan(ptr, elements_count));
  // }

  if (weights_are_buffer)
  {
    BufferDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    op->args_.AddObject("weights", absl::make_unique<BufferDescriptor>(std::move(desc)));
  }
  else
  {
    Texture2DDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.size = int2(kernel_x * kernel_y * kernel_z, dst_slices);
    desc.data = std::move(data);
    op->args_.AddObject("weights", absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

GPUOperation CreateDepthwiseConvolution2D(const DeviceInfo &device_info,
                                          const OperationDef &definition,
                                          const DepthwiseConvolution2DAttributes &attr);

GPUOperation
CreateDepthwiseConvolution2DDynamicWeights(const DeviceInfo &device_info,
                                           const OperationDef &definition,
                                           const DepthwiseConvolution2DAttributes &attr);

GPUOperation CreateDepthwiseConvolution3D(const DeviceInfo &device_info,
                                          const OperationDef &definition,
                                          const DepthwiseConvolution3DAttributes &attr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_H__
