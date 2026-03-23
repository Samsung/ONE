/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "DepthwiseConv2D.float.h"
#include "Assert.h"

#include "internal/Spatial.h"
#include "internal/Array.h"
#include "internal/Fused.h"
#include "internal/ActivationUtils.h"

#include <cstring> // 'memcpy'

namespace optimized_ops
{

// Implementation of float DepthwiseConv

template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
struct FloatDepthwiseConvKernel
{
};

// From optimized_ops.h in TensorFlow Lite
//
// Accumulates the effect of one row of the filter, on a segment of one row
// of the output, accessing the corresponding one row of the input.
template <bool kAllowStrided, int kFixedInputDepth, int kFixedDepthMultiplier>
void FloatDepthwiseConvAccumRow(int stride, int input_depth, int input_width,
                                const float *input_data, int pad_width, int depth_multiplier,
                                int filter_width, const float *filter_data, int out_x_buffer_start,
                                int out_x_buffer_end, int output_depth, float *acc_buffer)
{
  // Sanity check parameters. This is important in particular to ensure
  // that we keep the number of template instantiations minimal, so we don't
  // increase binary size unnecessarily.
  static_assert(kFixedDepthMultiplier || !kFixedInputDepth, "");
  static_assert(kFixedInputDepth || kAllowStrided, "");
  DCHECK(stride == 1 || kAllowStrided);
  if (kFixedInputDepth)
  {
    DCHECK_EQ(input_depth, kFixedInputDepth);
  }
  if (kFixedDepthMultiplier)
  {
    DCHECK_EQ(depth_multiplier, kFixedDepthMultiplier);
  }
  DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  const int input_ptr_increment = stride * input_depth;
  const float *filter_base_ptr = filter_data;
  for (int filter_x = 0; filter_x < filter_width; ++filter_x)
  {
    // For the current (filter_x, filter_y) point in the filter,
    // compute the boundaries of the corresponding output row segment.
    int out_x_loop_start_unclampled = 0;
    int out_x_loop_end_unclampled = 0;
    if (kAllowStrided)
    {
      if (stride == 2)
      {
        out_x_loop_start_unclampled = (pad_width - filter_x + 1) / 2;
        out_x_loop_end_unclampled = (pad_width + input_width - filter_x + 1) / 2;
      }
      else if (stride == 4)
      {
        out_x_loop_start_unclampled = (pad_width - filter_x + 3) / 4;
        out_x_loop_end_unclampled = (pad_width + input_width - filter_x + 3) / 4;
      }
      else
      {
        out_x_loop_start_unclampled = (pad_width - filter_x + stride - 1) / stride;
        out_x_loop_end_unclampled = (pad_width + input_width - filter_x + stride - 1) / stride;
      }
    }
    else
    {
      out_x_loop_start_unclampled = pad_width - filter_x;
      out_x_loop_end_unclampled = pad_width + input_width - filter_x;
    }
    // The kernel will have to iterate on the segment of the
    // output row that starts at out_x_loop_start and out_x_loop_end.
    const int out_x_loop_start = std::max(out_x_buffer_start, out_x_loop_start_unclampled);
    const int out_x_loop_end = std::min(out_x_buffer_end, out_x_loop_end_unclampled);

    float *acc_buffer_ptr = acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin = (out_x_loop_start * stride) - pad_width + filter_x;
    const float *input_ptr = input_data + in_x_origin * input_depth;
    const int num_output_pixels = out_x_loop_end - out_x_loop_start;
    FloatDepthwiseConvKernel<kAllowStrided, kFixedInputDepth, kFixedDepthMultiplier>::Run(
        num_output_pixels, input_depth, depth_multiplier, input_ptr, input_ptr_increment,
        filter_base_ptr, acc_buffer_ptr);
    filter_base_ptr += output_depth;
  }
}

// From optimized_ops.h in TensorFlow Lite
//
// generic fallback of FloatDepthwiseConvAccumRow, portable, non-templatized.
inline void FloatDepthwiseConvAccumRowGeneric(int stride, int input_depth, int input_width,
                                              const float *input_data, int pad_width,
                                              int depth_multiplier, int filter_width,
                                              const float *filter_data, int out_x_buffer_start,
                                              int out_x_buffer_end, int output_depth,
                                              float *acc_buffer)
{
  const float *filter_base_ptr = filter_data;
  for (int filter_x = 0; filter_x < filter_width; ++filter_x)
  {
    const int out_x_loop_start =
        std::max(out_x_buffer_start, (pad_width - filter_x + stride - 1) / stride);
    const int out_x_loop_end =
        std::min(out_x_buffer_end, (pad_width + input_width - filter_x + stride - 1) / stride);

    float *acc_buffer_ptr = acc_buffer + (out_x_loop_start - out_x_buffer_start) * output_depth;
    const int in_x_origin = (out_x_loop_start * stride) - pad_width + filter_x;
    const float *input_ptr = input_data + in_x_origin * input_depth;
    const int input_ptr_increment = (stride - 1) * input_depth;
    for (int out_x = out_x_loop_start; out_x < out_x_loop_end; out_x++)
    {
      const float *filter_ptr = filter_base_ptr;
      for (int ic = 0; ic < input_depth; ++ic)
      {
        const float input_val = *input_ptr++;
        for (int m = 0; m < depth_multiplier; m++)
        {
          const float filter_val = *filter_ptr++;
          *acc_buffer_ptr++ += filter_val * input_val;
        }
      }
      input_ptr += input_ptr_increment;
    }
    filter_base_ptr += output_depth;
  }
}

// From optimized_ops.h in TensorFlow Lite
//
// Initializes the accumulator buffer with bias values.
inline void DepthwiseConvInitAccBuffer(int num_output_pixels, int output_depth,
                                       const float *bias_data, float *acc_buffer)
{
  for (int i = 0; i < num_output_pixels; i++)
  {
    memcpy(acc_buffer + i * output_depth, bias_data, sizeof(acc_buffer[0]) * output_depth);
  }
}

// From optimized_ops.h in TensorFlow Lite
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float *input_data, const Dims<4> &input_dims, const float *filter_data,
                   const Dims<4> &filter_dims, const float *bias_data, const Dims<4> &bias_dims,
                   int stride_width, int stride_height, int pad_width, int pad_height,
                   int depth_multiplier, float *output_data, const Dims<4> &output_dims)
{
  static_assert(
      Ac == FusedActivationFunctionType::kNone || Ac == FusedActivationFunctionType::kRelu ||
          Ac == FusedActivationFunctionType::kRelu6 || Ac == FusedActivationFunctionType::kRelu1,
      "");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
#if 0 // TODO-NNRT : Check if assertion is needed, output depth some times not equal to input *
      // depthmultiplier
  DCHECK(output_depth == input_depth * depth_multiplier);
#endif

  static const int kAccBufferMaxSize = 1024;
  float acc_buffer[kAccBufferMaxSize];
  DCHECK_GE(kAccBufferMaxSize, output_depth);
  const int kOutputPixelsInAccBuffer = kAccBufferMaxSize / output_depth;
  const int kAccBufferActualSize = kOutputPixelsInAccBuffer * output_depth;
  DCHECK_LE(kOutputPixelsInAccBuffer * output_depth, kAccBufferActualSize);
  DCHECK_LE(kAccBufferActualSize, kAccBufferMaxSize);
  DCHECK_GE(kOutputPixelsInAccBuffer, 1);

  // row_accum_func will point to the core accumulation function to be used
  // for this DepthwiseConv op.
  auto *row_accum_func = FloatDepthwiseConvAccumRowGeneric;

  const int kMaxFixedDepthMultiplier = 16;
  int fixed_depth_multiplier = 0;
  if (depth_multiplier <= kMaxFixedDepthMultiplier)
  {
    fixed_depth_multiplier = depth_multiplier;
  }
  // kMaxUnrolling is the max number of output values that we aim to handle
  // in one unrolled iteration of the inner loop. For practical performance
  // reasons, it is limited by the number of available registers. We could
  // fine-tune it depending on the architecture, but that's not worth doing
  // since this whole code is not very optimized to begin with. The
  // present value reflects what's realistic on ARM 32bit NEON with 16 128-bit
  // vector registers.
  const int kMaxUnrolling = 8;
  int fixed_input_depth = 0;
  if (fixed_depth_multiplier && input_depth * fixed_depth_multiplier <= kMaxUnrolling)
  {
    fixed_input_depth = input_depth;
  }

  // Now that we have determined row_accum_func, we can start work.
  float *output_ptr = output_data;
  for (int b = 0; b < batches; ++b)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int filter_y_start = std::max(0, -in_y_origin);
      const int filter_y_end = std::min(filter_height, input_height - in_y_origin);
      for (int out_x_buffer_start = 0; out_x_buffer_start < output_width;
           out_x_buffer_start += kOutputPixelsInAccBuffer)
      {
        const int out_x_buffer_end =
            std::min(output_width, out_x_buffer_start + kOutputPixelsInAccBuffer);
        // We call a 'pixel' a group of activation that share all but the
        // 'depth'/'channel' coordinate. num_output_pixels is the number of
        // output pixels that we will accumulate in this loop iteration.
        const int num_output_pixels = out_x_buffer_end - out_x_buffer_start;
        // Initialize our local accumulator with the bias values, so we don't
        // have to add them later.
        DepthwiseConvInitAccBuffer(num_output_pixels, output_depth, bias_data, acc_buffer);
        // Accumulation loop. Most of the time should be spent in here.
        for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
        {
          const int in_y = in_y_origin + filter_y;
          row_accum_func(stride_width, input_depth, input_width,
                         input_data + in_y * input_dims.strides[2] + b * input_dims.strides[3],
                         pad_width, depth_multiplier, filter_width,
                         filter_data + filter_y * filter_dims.strides[2], out_x_buffer_start,
                         out_x_buffer_end, output_depth, acc_buffer);
        }
        // Finished accumulating. Now store to destination.
        const int num_output_values = output_depth * num_output_pixels;
        int i = 0;
        // Handle leftover values, one by one. This is very slow.
        for (; i < num_output_values; i++)
        {
          float acc = acc_buffer[i];
          if (Ac == FusedActivationFunctionType::kRelu)
          {
            acc = std::max(0.f, acc);
          }
          else if (Ac == FusedActivationFunctionType::kRelu6)
          {
            acc = std::max(0.f, std::min(6.f, acc));
          }
          else if (Ac == FusedActivationFunctionType::kRelu1)
          {
            acc = std::max(-1.f, std::min(1.f, acc));
          }
          *output_ptr++ = acc;
        }
      }
    }
  }
}

} // namespace optimized_ops

#define ANDROID_NN_DEPTHWISE_CONV_PARAMETERS                  \
  uint32_t height = getSizeOfDimension(inputShape, 1);        \
  uint32_t width = getSizeOfDimension(inputShape, 2);         \
  uint32_t filterHeight = getSizeOfDimension(filterShape, 1); \
  uint32_t filterWidth = getSizeOfDimension(filterShape, 2);  \
  uint32_t outHeight = getSizeOfDimension(outputShape, 1);    \
  uint32_t outWidth = getSizeOfDimension(outputShape, 2);     \
                                                              \
  uint32_t paddingHeight = (uint32_t)padding_top;             \
  uint32_t paddingWidth = (uint32_t)padding_left;

bool depthwiseConvFloat32(const float *inputData, const Shape &inputShape, const float *filterData,
                          const Shape &filterShape, const float *biasData, const Shape &biasShape,
                          int32_t padding_left, int32_t padding_right, int32_t padding_top,
                          int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation, float *outputData,
                          const Shape &outputShape)
{

  ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

#define ANDROID_NN_DEPTHWISE_CONV(activation)                                                 \
  optimized_ops::DepthwiseConv<FusedActivationFunctionType::activation>(                      \
      inputData, convertShapeToDims(inputShape), filterData, convertShapeToDims(filterShape), \
      biasData, convertShapeToDims(biasShape), stride_width, stride_height, paddingWidth,     \
      paddingHeight, depth_multiplier, outputData, convertShapeToDims(outputShape))

  ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_DEPTHWISE_CONV)
#undef ANDROID_NN_DEPTHWISE_CONV

  return true;
}
