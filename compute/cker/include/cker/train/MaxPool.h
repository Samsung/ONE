/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_TRAIN_MAX_POOL_H__
#define __NNFW_CKER_TRAIN_MAX_POOL_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/neon/neon_check.h"
#include "cker/eigen/Utils.h"

#include <Eigen/Core>

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T> void MaxPool(const PoolParams &, const Shape &, const T *, const Shape &, T *)
{
  static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                "cker::MaxPool : This function supports only integer or floating point");
  throw std::runtime_error("cker::MaxPool : Unsupported data type");
}

template <>
void MaxPool<float>(const PoolParams &params, const Shape &input_shape, const float *input_data,
                    const Shape &output_shape, float *output_data)
{
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // Prefill the output to minimum representable float value
  out_mat.setConstant(std::numeric_limits<float>::lowest());
  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < input_height; ++h)
    {
      for (int w = 0; w < input_width; ++w)
      {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start =
          (hpad < params.filter_height) ? 0 : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start =
          (wpad < params.filter_width) ? 0 : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph)
        {
          for (int pw = w_start; pw < w_end; ++pw)
          {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) =
              out_mat.col(out_offset)
                .cwiseMax(in_mat.col(NodeOffset(b, h, w, input_height, input_width)));
          }
        }
      }
    }
  }
  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i], params.float_activation_min,
                                                  params.float_activation_max);
  }
}

template <>
void MaxPool<uint8_t>(const PoolParams &params, const Shape &input_shape, const uint8_t *input_data,
                      const Shape &output_shape, uint8_t *output_data)
{

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  assert(params.quantized_activation_min <= params.quantized_activation_max);
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint8_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch)
  {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth; depth_base += kPoolingAccTrancheSize)
    {
      const int tranche_depth = std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y)
      {
        for (int out_x = 0; out_x < output_width; ++out_x)
        {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8_t *input_ptr =
            input_data + depth_base +
            depth * (in_x_origin + input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++)
          {
            const uint8_t *input_row_ptr = input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++)
            {
              const uint8_t *input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16)
              {
                uint8x16_t acc_reg = vld1q_u8(acc + channel);
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_u8(acc_reg, input_reg);
                vst1q_u8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8)
              {
                uint8x8_t acc_reg = vld1_u8(acc + channel);
                uint8x8_t input_reg = vld1_u8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_u8(acc_reg, input_reg);
                vst1_u8(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel)
              {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          uint8_t *output_ptr = output_data + Offset(output_shape, batch, out_y, out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16)
          {
            uint8x16_t a = vld1q_u8(acc + channel);
            a = vminq_u8(a, vdupq_n_u8(params.quantized_activation_max));
            a = vmaxq_u8(a, vdupq_n_u8(params.quantized_activation_min));
            vst1q_u8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8)
          {
            uint8x8_t a = vld1_u8(acc + channel);
            a = vmin_u8(a, vdup_n_u8(params.quantized_activation_max));
            a = vmax_u8(a, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, a);
          }
#endif
          for (; channel < tranche_depth; ++channel)
          {
            uint8_t a = acc[channel];
            a = std::max<uint8_t>(a, params.quantized_activation_min);
            a = std::min<uint8_t>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8_t>(a);
          }
        }
      }
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_MAX_POOL_H__
