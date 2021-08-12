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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_UTIL_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_UTIL_H__

#include <string>

#include "absl/types/span.h"
#include "OpenclWrapper.h"
#include "DataType.h"
#include "InternalTensor.h"
#include "Status.h"
#include "Types.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
// Calculates correct X coordinate when stride != 1 and batch != 1 for layouts
// with B after W (for example HWBC4) and WB stored in one axis of GPU
// resources.
std::string GetXStrideCorrected(const std::string &src_x, const std::string &batch_size,
                                const std::string &stride_x, const std::string &padding_x);

// @param n must be non negative
// @param divisor must be greater than zero
template <typename T, typename N> T DivideRoundUp(T n, N divisor)
{
  const T div = static_cast<T>(divisor);
  const T q = n / div;
  return n % div == 0 ? q : q + 1;
}

template <> inline uint3 DivideRoundUp(uint3 n, uint3 divisor)
{
  return uint3(DivideRoundUp(n.x, divisor.x), DivideRoundUp(n.y, divisor.y),
               DivideRoundUp(n.z, divisor.z));
}

// @param number or its components must be greater than zero
// @param n must be greater than zero
template <typename T, typename N> T AlignByN(T number, N n) { return DivideRoundUp(number, n) * n; }

std::string CLErrorCodeToString(cl_int error_code);

int ChannelTypeToSizeInBytes(cl_channel_type type);

template <DataType S, typename T>
void CopyLinearFLT4(const InternalTensor<Linear, S> &src, absl::Span<T> dst)
{
  const int dst_depth = dst.size();
  for (int d = 0; d < dst_depth; ++d)
  {
    T val;
    for (int i = 0; i < 4; ++i)
    {
      const int dst_ch = d * 4 + i;
      val[i] = dst_ch >= src.shape.v ? 0.0f : src.data[dst_ch];
    }
    dst[d] = val;
  }
}

absl::Status CreateCLBuffer(cl_context context, int size_in_bytes, bool read_only, void *data,
                            cl_mem *result);

cl_channel_type DataTypeToChannelType(DataType type, bool normalized = false);
absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void *data, cl_mem *result);

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(const InternalTensor<OHWI, S> &weights, int out_group_size,
                                      absl::Span<T> dst)
{
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d)
  {
    for (int y = 0; y < weights.shape.h; ++y)
    {
      for (int x = 0; x < weights.shape.w; ++x)
      {
        for (int s = 0; s < src_slices; ++s)
        {
          for (int d_group = 0; d_group < out_group_size; ++d_group)
          {
            for (int j = 0; j < 4; ++j)
            {
              T filter;
              for (int i = 0; i < 4; ++i)
              {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * out_group_size + d_group) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o)
                {
                  const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filter[i] = weights.data[f_index];
                }
                else
                {
                  filter[i] = 0.0f;
                }
              }
              dst[counter++] = filter;
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToODHWIOGroupI4O4(const InternalTensor<OHWDI, S> &weights, int out_group_size,
                                       absl::Span<T> dst)
{
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d)
  {
    for (int z = 0; z < weights.shape.d; ++z)
    {
      for (int y = 0; y < weights.shape.h; ++y)
      {
        for (int x = 0; x < weights.shape.w; ++x)
        {
          for (int s = 0; s < src_slices; ++s)
          {
            for (int d_group = 0; d_group < out_group_size; ++d_group)
            {
              for (int j = 0; j < 4; ++j)
              {
                T filter;
                for (int i = 0; i < 4; ++i)
                {
                  const int s_ch = s * 4 + j;
                  const int d_ch = (d * out_group_size + d_group) * 4 + i;
                  if (s_ch < weights.shape.i && d_ch < weights.shape.o)
                  {
                    const int f_index = weights.shape.LinearIndex({d_ch, y, x, z, s_ch});
                    filter[i] = weights.data[f_index];
                  }
                  else
                  {
                    filter[i] = 0.0f;
                  }
                }
                dst[counter++] = filter;
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToI4HWIOOGroupO4(const InternalTensor<OHWI, S> &weights, int out_group_size,
                                      absl::Span<T> dst)
{
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int j = 0; j < 4; ++j)
  {
    for (int y = 0; y < weights.shape.h; ++y)
    {
      for (int x = 0; x < weights.shape.w; ++x)
      {
        for (int s = 0; s < src_slices; ++s)
        {
          for (int d = 0; d < dst_groups; ++d)
          {
            for (int d_group = 0; d_group < out_group_size; ++d_group)
            {
              T filter;
              for (int i = 0; i < 4; ++i)
              {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * out_group_size + d_group) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o)
                {
                  const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filter[i] = weights.data[f_index];
                }
                else
                {
                  filter[i] = 0.0f;
                }
              }
              dst[counter++] = filter;
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToI4DHWIOOGroupO4(const InternalTensor<OHWDI, S> &weights, int out_group_size,
                                       absl::Span<T> dst)
{
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int j = 0; j < 4; ++j)
  {
    for (int z = 0; z < weights.shape.d; ++z)
    {
      for (int y = 0; y < weights.shape.h; ++y)
      {
        for (int x = 0; x < weights.shape.w; ++x)
        {
          for (int s = 0; s < src_slices; ++s)
          {
            for (int d = 0; d < dst_groups; ++d)
            {
              for (int d_group = 0; d_group < out_group_size; ++d_group)
              {
                T filter;
                for (int i = 0; i < 4; ++i)
                {
                  const int s_ch = s * 4 + j;
                  const int d_ch = (d * out_group_size + d_group) * 4 + i;
                  if (s_ch < weights.shape.i && d_ch < weights.shape.o)
                  {
                    const int f_index = weights.shape.LinearIndex({d_ch, y, x, z, s_ch});
                    filter[i] = weights.data[f_index];
                  }
                  else
                  {
                    filter[i] = 0.0f;
                  }
                }
                dst[counter++] = filter;
              }
            }
          }
        }
      }
    }
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_UTIL_H__
