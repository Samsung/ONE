/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_OPTIMIZED_OPTIMIZED_UTILS_H__
#define __NNFW_CKER_OPTIMIZED_OPTIMIZED_UTILS_H__

#include "cker/Types.h"
#include "cker/Shape.h"

#include <stdexcept>

namespace nnfw
{
namespace cker
{
namespace optimized
{

template <typename T>
inline void ExtractPatchIntoBufferColumn(const Shape &input_shape, int w, int h, int b, int kheight,
                                         int kwidth, int stride_width, int stride_height,
                                         int pad_width, int pad_height, int in_width, int in_height,
                                         int in_depth, int single_buffer_length, int buffer_id,
                                         const T *in_data, T *conv_buffer_data, uint8_t zero_byte)
{
  assert(input_shape.DimensionsCount() == 4);
  // This chunk of code reshapes all the inputs corresponding to
  // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
  const int kwidth_times_indepth = kwidth * in_depth;
  const int inwidth_times_indepth = in_width * in_depth;
  const int ih_ungated_start = h * stride_height - pad_height;
  const int ih_ungated_end = (ih_ungated_start + kheight);
  const int ih_end = std::min(ih_ungated_end, in_height);
  const int iw_ungated_start = w * stride_width - pad_width;
  const int iw_ungated_end = (iw_ungated_start + kwidth);
  const int iw_end = std::min(iw_ungated_end, in_width);
  // If the patch is off the edge of the input image, skip writing those rows
  // and columns from the patch into the output array.
  const int h_offset = std::max(0, -ih_ungated_start);
  const int w_offset = std::max(0, -iw_ungated_start);
  const int ih_start = std::max(0, ih_ungated_start);
  const int iw_start = std::max(0, iw_ungated_start);
  const int single_row_num = std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
  const int output_row_offset = (buffer_id * single_buffer_length);
  int out_offset = output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
  int in_offset = Offset(input_shape, b, ih_start, iw_start, 0);

  // Express all of the calculations as padding around the input patch.
  const int top_padding = h_offset;
  const int bottom_padding = (ih_ungated_end - ih_end);
  const int left_padding = w_offset;
  const int right_padding = (iw_ungated_end - iw_end);
  assert(single_row_num == ((kwidth - (left_padding + right_padding)) * in_depth));

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0)
  {
    const int top_row_elements = (top_padding * kwidth * in_depth);
    memset(conv_buffer_data + output_row_offset, zero_byte, (top_row_elements * sizeof(T)));
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if ((left_padding == 0) && (right_padding == 0))
  {
    for (int ih = ih_start; ih < ih_end; ++ih)
    {
      memcpy(conv_buffer_data + out_offset, in_data + in_offset, single_row_num * sizeof(T));
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  }
  else
  {
    for (int ih = ih_start; ih < ih_end; ++ih)
    {
      if (left_padding > 0)
      {
        const int left_start = (out_offset - (left_padding * in_depth));
        memset(conv_buffer_data + left_start, zero_byte, (left_padding * in_depth * sizeof(T)));
      }
      memcpy(conv_buffer_data + out_offset, in_data + in_offset, single_row_num * sizeof(T));
      if (right_padding > 0)
      {
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + right_start, zero_byte, (right_padding * in_depth * sizeof(T)));
      }
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0)
  {
    const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
    const int bottom_start =
        output_row_offset + ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
    memset(conv_buffer_data + bottom_start, zero_byte, (bottom_row_elements * sizeof(T)));
  }
}

template <typename T>
void DilatedIm2col(const ConvParams &params, uint8_t zero_byte, const Shape &input_shape,
                   const T *input_data, const Shape &filter_shape, const Shape &output_shape,
                   T *im2col_data)
{
  (void)params;
  (void)zero_byte;
  (void)input_shape;
  (void)input_data;
  (void)filter_shape;
  (void)output_shape;
  (void)im2col_data;
  throw std::runtime_error{"NYI: cker DilatedIm2col"};
}

template <typename T>
void Im2col(const ConvParams &params, int kheight, int kwidth, uint8_t zero_byte,
            const Shape &input_shape, const T *input_data, const Shape &output_shape,
            T *output_data)
{
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < output_height; ++h)
    {
      for (int w = 0; w < output_width; ++w)
      {
        ExtractPatchIntoBufferColumn(input_shape, w, h, b, kheight, kwidth, stride_width,
                                     stride_height, pad_width, pad_height, input_width,
                                     input_height, input_depth, output_depth, buffer_id, input_data,
                                     output_data, zero_byte);
        ++buffer_id;
      }
    }
  }
}

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_OPTIMIZED_UTILS_H__
