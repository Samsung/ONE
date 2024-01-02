/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_DEPTHWISECONV_H__
#define __NNFW_CKER_TRAIN_OPERATION_DEPTHWISECONV_H__

#include "cker/operation/Helper/Tensor.h"
#include "cker/eigen/eigen_backward_spatial_convolutions.h"
#include "cker/eigen/EigenSupport.h"
#include "cker/eigen/Utils.h"
#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{
namespace train
{

// From tensorflow/core/kernels/depthwise_conv_op.h
// From tensorflow/core/kernels/depthwise_conv_grad_op.cc
// template <typename T>
// struct LaunchDepthwiseConvBackpropInputOp<<T> {
//   typedef typename Eigen::internal::packet_traits<T>::type Packet;

//   void operator()(const DepthwiseArgs &args,
//                   out_depth, filter_rows, filter_cols,
//                   const T* out_backprop, const T* depthwise_filter,
//                   T* in_backprop) {
//     static const int64_6 kPacketSize = (sizeof(Packet) / sizeof(T));

//     // Pad 'depthwise_filter' to vector register width (if needed).
//     const bool pad_filter = (out_depth % kPacketSize) == 0 ? false : true;
//     std::vector<T> padded_filter_data;
//     if (pad_filter) {
//       // Allocate space for padded filter.
//       const int64_t filter_spatial_size = filter_rows * filter_cols;
//       const int64_t padded_filter_inner_dim_size =
//         ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
//       padded_filter_data.assign({filter_spatial_size, padded_filter_inner_dim_size});
//       // Write out padded filter.
//       functor::DepthwiseFilterPadOp<T>()(
//         args, depthwise_filter, padded_filter_data.data());
//     }
//     const T* filter_data =
//       pad_filter ? padded_filter_data.data() : depthwise_filter;

//     // Computes one shard of depthwise conv2d backprop input.

//   }
// }

template <typename T>
static void
DepthwiseConvBackpropInputReference(int batch, int in_rows, int in_cols, int in_depth, int out_rows,
                                    int out_cols, int out_depth, int stride, int depth_multiplier,
                                    int filter_rows, int filter_cols, int pad_rows, int pad_cols,
                                    const T *out_backprop, const T *filter, T *in_backprop)
{
  // Naive for loop as a reference point without concerns about performance.
  for (int b = 0; b < batch; ++b)
  {
    for (int in_r = 0; in_r < in_rows; ++in_r)
    {
      for (int in_c = 0; in_c < in_cols; ++in_c)
      {
        for (int in_d = 0; in_d < in_depth; ++in_d)
        {
          T sum = 0;
          const int out_d_start = in_d * depth_multiplier;
          const int out_d_end = out_d_start + depth_multiplier;

          for (int out_d = out_d_start; out_d < out_d_end; ++out_d)
          {
            const int out_r_start = std::max(0, (in_r - filter_rows + pad_rows + stride) / stride);
            const int out_r_end = std::min(out_rows - 1, (in_r + pad_rows) / stride);

            for (int out_r = out_r_start; out_r <= out_r_end; ++out_r)
            {
              const int out_c_start =
                std::max(0, (in_c - filter_cols + pad_cols + stride) / stride);
              const int out_c_end = std::min(out_cols - 1, (in_c + pad_cols) / stride);

              for (int out_c = out_c_start; out_c <= out_c_end; ++out_c)
              {
                int f_r = in_r + pad_rows - out_r * stride;
                int f_c = in_c + pad_cols - out_c * stride;
                int filter_dm = out_d - out_d_start;
                int out_backprop_offset =
                  out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
                int filter_offset =
                  filter_dm + depth_multiplier * (in_d + in_depth * (f_c + filter_cols * f_r));
                sum += out_backprop[out_backprop_offset] * filter[filter_offset];
              }
            }
          }

          int in_backprop_offset = in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
          in_backprop[in_backprop_offset] = sum;
        }
      }
    }
  }
}

template <typename T>
static void DepthwiseConvBackpropFilterReference(int batch, int in_rows, int in_cols, int in_depth,
                                                 int out_rows, int out_cols, int out_depth,
                                                 int stride, int depth_multiplier, int filter_rows,
                                                 int filter_cols, int pad_rows, int pad_cols,
                                                 const T *out_backprop, const T *input,
                                                 T *filter_backprop)
{
  int num_filter_backprop = filter_rows * filter_cols * in_depth * depth_multiplier;
  memset(filter_backprop, 0, num_filter_backprop * sizeof(T));
  // Naive for loop as a reference point without concerns about performance.
  for (int b = 0; b < batch; ++b)
  {
    for (int out_r = 0; out_r < out_rows; ++out_r)
    {
      for (int out_c = 0; out_c < out_cols; ++out_c)
      {
        for (int out_d = 0; out_d < out_depth; ++out_d)
        {
          const int in_d = out_d / depth_multiplier;
          const int dm = out_d % depth_multiplier;
          const int in_r_start = out_r * stride - pad_rows;
          const int in_c_start = out_c * stride - pad_cols;

          for (int f_r = 0; f_r < filter_rows; ++f_r)
          {
            for (int f_c = 0; f_c < filter_cols; ++f_c)
            {
              const int in_r = in_r_start + f_r;
              const int in_c = in_c_start + f_c;

              if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols)
              {
                int out_backprop_offset =
                  out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
                int input_offset = in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
                int filter_backprop_offset =
                  dm + depth_multiplier * (in_d + in_depth * (f_c + filter_cols * f_r));
                filter_backprop[filter_backprop_offset] +=
                  input[input_offset] * out_backprop[out_backprop_offset];
              }
            }
          }
        }
      }
    }
  }
}

inline void DepthwiseConvInputGrad(const DepthwiseConvParams &params, const Shape &incoming_shape,
                                   const float *incoming_data, const Shape &filter_shape,
                                   const float *filter_data, const Shape &grad_shape,
                                   float *grad_data)
{
  // LaunchDepthwiseConvBackpropInputOp<float>()(
  //   /* To be filled */
  // )
  const int batch = MatchingDim(incoming_shape, 0, grad_shape, 0);
  const int input_depth = grad_shape.Dims(3);
  const int output_depth = incoming_shape.Dims(3);
  const int incoming_height = incoming_shape.Dims(1);
  const int incoming_width = incoming_shape.Dims(2);
  const int grad_height = grad_shape.Dims(1);
  const int grad_width = grad_shape.Dims(2);
  assert(params.stride_height == params.stride_width);
  const int stride = params.stride_height;
  const int depth_multiplier = params.depth_multiplier;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int pad_height = params.padding_values.height;
  const int pad_width = params.padding_values.width;

  DepthwiseConvBackpropInputReference<float>(
    batch, grad_height, grad_width, input_depth, incoming_height, incoming_width, output_depth,
    stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
    filter_data, grad_data);
}

inline void DepthwiseConvFilterGrad(const DepthwiseConvParams &params, const Shape &incoming_shape,
                                    const float *incoming_data, const Shape &input_shape,
                                    const float *input_data, const Shape &filter_grad_shape,
                                    float *filter_grad_data)
{
  const int batch = MatchingDim(incoming_shape, 0, input_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = incoming_shape.Dims(3);
  const int incoming_height = incoming_shape.Dims(1);
  const int incoming_width = incoming_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  assert(params.stride_height == params.stride_width);
  const int stride = params.stride_height;
  const int depth_multiplier = params.depth_multiplier;
  const int filter_height = filter_grad_shape.Dims(1);
  const int filter_width = filter_grad_shape.Dims(2);
  const int pad_height = params.padding_values.height;
  const int pad_width = params.padding_values.width;

  DepthwiseConvBackpropFilterReference(batch, input_height, input_width, input_depth,
                                       incoming_height, incoming_width, output_depth, stride,
                                       depth_multiplier, filter_height, filter_width, pad_height,
                                       pad_width, incoming_data, input_data, filter_grad_data);
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_DEPTHWISECONV_H__
