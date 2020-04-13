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

#include "Conv2D.float.h"

#include "internal/Spatial.h"
#include "internal/Array.h"
#include "internal/Matrix.h"
#include "internal/Fused.h"
#include "internal/GEMM.h"
#include "internal/ActivationUtils.h"

// From optimized_ops.h in TensorFlow Lite
template <typename T>
inline void ExtractPatchIntoBufferColumn(const Dims<4> &input_dims, int w, int h, int b,
                                         int kheight, int kwidth, int stride_width,
                                         int stride_height, int pad_width, int pad_height,
                                         int in_width, int in_height, int in_depth,
                                         int single_buffer_length, int buffer_id, const T *in_data,
                                         T *conv_buffer_data, uint8 byte_zero)
{
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
  int in_offset = Offset(input_dims, 0, iw_start, ih_start, b);

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
    memset(conv_buffer_data + output_row_offset, byte_zero, (top_row_elements * sizeof(T)));
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
        memset(conv_buffer_data + left_start, byte_zero, (left_padding * in_depth * sizeof(T)));
      }
      memcpy(conv_buffer_data + out_offset, in_data + in_offset, single_row_num * sizeof(T));
      if (right_padding > 0)
      {
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + right_start, byte_zero, (right_padding * in_depth * sizeof(T)));
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
    memset(conv_buffer_data + bottom_start, byte_zero, (bottom_row_elements * sizeof(T)));
  }
}

template <typename T>
void Im2col(const T *input_data, const Dims<4> &input_dims, int stride_width, int stride_height,
            int pad_width, int pad_height, int kheight, int kwidth, uint8 byte_zero, T *output_data,
            const Dims<4> &output_dims)
{
  DCHECK(IsPackedWithoutStrides(input_dims));
  DCHECK(IsPackedWithoutStrides(output_dims));
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < output_height; ++h)
    {
      for (int w = 0; w < output_width; ++w)
      {
        ExtractPatchIntoBufferColumn(input_dims, w, h, b, kheight, kwidth, stride_width,
                                     stride_height, pad_width, pad_height, input_width,
                                     input_height, input_depth, output_depth, buffer_id, input_data,
                                     output_data, byte_zero);
        ++buffer_id;
      }
    }
  }
}

// From optimized_ops.h in TensorFlow Lite
template <FusedActivationFunctionType Ac>
void Conv(const float *input_data, const Dims<4> &input_dims, const float *filter_data,
          const Dims<4> &filter_dims, const float *bias_data, const Dims<4> &bias_dims,
          int stride_width, int stride_height, int pad_width, int pad_height, float *output_data,
          const Dims<4> &output_dims, float *im2col_data, const Dims<4> &im2col_dims)
{
  (void)im2col_data;
  (void)im2col_dims;

  const float *gemm_input_data = nullptr;
  const Dims<4> *gemm_input_dims = nullptr;
  const int filter_width = ArraySize(filter_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const bool need_im2col =
      stride_width != 1 || stride_height != 1 || filter_width != 1 || filter_height != 1;
  if (need_im2col)
  {
    DCHECK(im2col_data);
    Im2col(input_data, input_dims, stride_width, stride_height, pad_width, pad_height,
           filter_height, filter_width, 0, im2col_data, im2col_dims);
    gemm_input_data = im2col_data;
    gemm_input_dims = &im2col_dims;
  }
  else
  {
#if 0 // TODO-NNRT : Check if it needs, 'im2col_data' seems to be always not null.
    DCHECK(!im2col_data);
#endif
    gemm_input_data = input_data;
    gemm_input_dims = &input_dims;
  }

  const auto im2col_matrix_map = MapAsMatrixWithFirstDimAsRows(gemm_input_data, *gemm_input_dims);
  const auto filter_matrix_map = MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), im2col_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data, output_dims);
}

// If possible we will use this static buffer for the tensor.
static constexpr int kStaticBufferSize = 1605632;
static char static_scratch_buffer[kStaticBufferSize];

#define ANDROID_NN_CONV_PARAMETERS(Type)                                      \
  uint32_t height = getSizeOfDimension(inputShape, 1);                        \
  uint32_t width = getSizeOfDimension(inputShape, 2);                         \
  uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
  uint32_t filterWidth = getSizeOfDimension(filterShape, 2);                  \
  uint32_t outHeight = getSizeOfDimension(outputShape, 1);                    \
  uint32_t outWidth = getSizeOfDimension(outputShape, 2);                     \
  uint32_t inDepth = getSizeOfDimension(inputShape, 3);                       \
                                                                              \
  uint32_t paddingHeight = (uint32_t)padding_top;                             \
  uint32_t paddingWidth = (uint32_t)padding_left;                             \
                                                                              \
  Dims<4> im2colDim;                                                          \
  im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);               \
  im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);               \
  im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);               \
  im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;             \
                                                                              \
  im2colDim.strides[0] = 1;                                                   \
  for (int i = 1; i < 4; i++)                                                 \
  {                                                                           \
    im2colDim.strides[i] = im2colDim.strides[i - 1] * im2colDim.sizes[i - 1]; \
  }                                                                           \
                                                                              \
  Type *im2colData = nullptr;                                                 \
  int im2colByteSize = sizeof(Type);                                          \
  for (int i = 0; i < 4; i++)                                                 \
  {                                                                           \
    im2colByteSize *= im2colDim.sizes[i];                                     \
  }                                                                           \
  if (im2colByteSize <= kStaticBufferSize)                                    \
  {                                                                           \
    im2colData = reinterpret_cast<Type *>(static_scratch_buffer);             \
  }                                                                           \
  else                                                                        \
  {                                                                           \
    im2colData = new (std::nothrow) Type[im2colByteSize / sizeof(Type)];      \
  }

bool convFloat32(const float *inputData, const Shape &inputShape, const float *filterData,
                 const Shape &filterShape, const float *biasData, const Shape &biasShape,
                 int32_t padding_left, int32_t padding_right, int32_t padding_top,
                 int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                 int32_t activation, float *outputData, const Shape &outputShape)
{

  ANDROID_NN_CONV_PARAMETERS(float)

#define ANDROID_NN_CONV(activation)                                                           \
  Conv<FusedActivationFunctionType::activation>(                                              \
      inputData, convertShapeToDims(inputShape), filterData, convertShapeToDims(filterShape), \
      biasData, convertShapeToDims(biasShape), stride_width, stride_height, paddingWidth,     \
      paddingHeight, outputData, convertShapeToDims(outputShape), im2colData, im2colDim)

  ANDROID_NN_MACRO_DISPATCH_WITH_DELETE(ANDROID_NN_CONV)
#undef ANDROID_NN_CONV

  if (im2colByteSize > kStaticBufferSize)
  {
    delete[] im2colData;
  }
  return true;
}
