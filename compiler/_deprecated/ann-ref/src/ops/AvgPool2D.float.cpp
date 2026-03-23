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

#include "AvgPool2D.float.h"

#include "internal/Array.h"
#include "internal/Matrix.h"
#include "internal/FeatureMap.h"
#include "internal/Fused.h"
#include "internal/ActivationUtils.h"

// From optimized_ops.h in TensorFlow Lite
template <FusedActivationFunctionType Ac>
void AveragePool(const float *input_data, const Dims<4> &input_dims, int stride_width,
                 int stride_height, int pad_width, int pad_height, int kwidth, int kheight,
                 float *output_data, const Dims<4> &output_dims)
{
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
  // TODO: get rid of the dynamic memory allocation here!
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < input_height; ++h)
    {
      for (int w = 0; w < input_width; ++w)
      {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + pad_height;
        int wpad = w + pad_width;
        int h_start = (hpad < kheight) ? 0 : (hpad - kheight) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < kwidth) ? 0 : (wpad - kwidth) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph)
        {
          for (int pw = w_start; pw < w_end; ++pw)
          {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) += in_mat.col(NodeOffset(b, h, w, input_height, input_width));
            out_count(out_offset)++;
          }
        }
      }
    }
  }
  // Divide the output by the actual number of elements being averaged over
  DCHECK_GT(out_count.minCoeff(), 0);
  out_mat.array().rowwise() /= out_count.transpose().array();

  for (int b = 0; b < batches; ++b)
  {
    for (int y = 0; y < output_height; ++y)
    {
      for (int x = 0; x < output_width; ++x)
      {
        for (int c = 0; c < depth; ++c)
        {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunction<Ac>(output_data[Offset(output_dims, c, x, y, b)]);
        }
      }
    }
  }
}

#define ANDROID_NN_POOLING_PARAMETERS                      \
  uint32_t height = getSizeOfDimension(inputShape, 1);     \
  uint32_t width = getSizeOfDimension(inputShape, 2);      \
  uint32_t outHeight = getSizeOfDimension(outputShape, 1); \
  uint32_t outWidth = getSizeOfDimension(outputShape, 2);  \
                                                           \
  uint32_t paddingHeight = (uint32_t)padding_top;          \
  uint32_t paddingWidth = (uint32_t)padding_left;

bool averagePoolFloat32(const float *inputData, const Shape &inputShape, int32_t padding_left,
                        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                        int32_t stride_width, int32_t stride_height, int32_t filter_width,
                        int32_t filter_height, int32_t activation, float *outputData,
                        const Shape &outputShape)
{

  ANDROID_NN_POOLING_PARAMETERS

#define ANDROID_NN_AVERAGE_POOL(activation)                                                 \
  AveragePool<FusedActivationFunctionType::activation>(                                     \
      inputData, convertShapeToDims(inputShape), stride_width, stride_height, paddingWidth, \
      paddingHeight, filter_width, filter_height, outputData, convertShapeToDims(outputShape))

  ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_AVERAGE_POOL)
#undef ANDROID_NN_AVERAGE_POOL

  return true;
}

#undef ANDROID_NN_POOLING_PARAMETERS
