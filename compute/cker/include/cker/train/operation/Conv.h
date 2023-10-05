/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_CONV_H__
#define __NNFW_CKER_TRAIN_OPERATION_CONV_H__

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

// From tensorflow/core/kernels/conv_2d.h
namespace functor
{

template <typename Device, typename T> struct SpatialConvolutionBackwardInputFunc
{
  void operator()(const Device &d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward, Eigen::DenseIndex col_stride,
                  Eigen::DenseIndex row_stride, Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation)
  {
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
      filter, output_backward, input_backward.dimension(2), input_backward.dimension(1), col_stride,
      row_stride, col_dilation, row_dilation);
  }
};

template <typename Device, typename T> struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc
{
  void operator()(const Device &d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward, Eigen::DenseIndex padded_cols,
                  Eigen::DenseIndex padded_rows, Eigen::DenseIndex col_stride,
                  Eigen::DenseIndex row_stride, Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation, Eigen::DenseIndex pad_left,
                  Eigen::DenseIndex pad_top)
  {
    // We have to slice the result of a spatial convolution backward
    // input, before assigning it to the `input_backward` to remove padding.
    //
    // TODO(ezhulenev): Pass explicit paddings to Eigen and do not materialize
    // intermediate result in memory before slicing.
    input_backward.device(d) =
      Eigen::SpatialConvolutionBackwardInput(filter, output_backward, padded_cols, padded_rows,
                                             col_stride, row_stride, col_dilation, row_dilation)
        .eval()
        .slice(Eigen::DSizes<Eigen::DenseIndex, 4>{0, pad_left, pad_top, 0},
               input_backward.dimensions());
  }
};

} // namespace functor

// From tensorflow/core/kernels/conv_grad_input_ops.h
// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU
// and GPU (for int32 only).
template <typename Device, typename T> struct LaunchConv2DBackpropInputOpImpl
{
  void operator()(const Device &d, const T *out_backprop_data, int batches, int out_backprop_height,
                  int out_backprop_width, int out_backprop_depth, const T *filter_data,
                  int filter_height, int filter_width, int row_dilation, int col_dilation,
                  int row_stride /* H */, int col_stride /* W */, const PaddingType &padding_type,
                  int padding_top, int padding_bottom, int padding_left, int padding_right,
                  T *in_backprop_data, int in_backprop_height, int in_backprop_width,
                  int in_backprop_depth)
  {
    // WARNING: Need to swap row/col, padding_top/padding_left, and
    // padding_bottom/padding_right when calling Eigen. Eigen expects tensors
    // in NWHC format, but Tensorflow uses NHWC.

    eigen_support::EigenTensor in_backprop_t(in_backprop_data, batches, in_backprop_height,
                                             in_backprop_width, in_backprop_depth);
    eigen_support::ConstEigenTensor filter_t(filter_data, filter_height, filter_width,
                                             in_backprop_depth, out_backprop_depth);
    eigen_support::ConstEigenTensor out_backprop_t(out_backprop_data, batches, out_backprop_height,
                                                   out_backprop_width, out_backprop_depth);

    if (padding_type != PaddingType::kNone /* EXPLICIT */)
    {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward input will infer correct forward paddings from input tensors.
      functor::SpatialConvolutionBackwardInputFunc<Device, T>()(
        d, in_backprop_t, filter_t, out_backprop_t, col_stride, row_stride, col_dilation,
        row_dilation);
    }
    else
    {
      functor::SpatialConvolutionBackwardInputWithExplicitPaddingFunc<Device, T>()(
        d, in_backprop_t, filter_t, out_backprop_t,
        in_backprop_t.dimension(2) + (padding_left + padding_right),
        in_backprop_t.dimension(1) + (padding_top + padding_bottom), col_stride, row_stride,
        col_dilation, row_dilation, padding_top, padding_left);
    }
  }
};

// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU.
template <typename T> struct LaunchConv2DBackpropInputOp
{
  void operator()(const T *out_backprop_data, int batches, int out_backprop_height,
                  int out_backprop_width, int out_backprop_depth, const T *filter_data,
                  int filter_height, int filter_width, int row_dilation, int col_dilation,
                  int row_stride /* H */, int col_stride /* W */, const PaddingType &padding_type,
                  int padding_top, int padding_bottom, int padding_left, int padding_right,
                  T *in_backprop_data, int in_backprop_height, int in_backprop_width,
                  int in_backprop_depth)
  {
    LaunchConv2DBackpropInputOpImpl<Eigen::ThreadPoolDevice, T> launcher;
    launcher(*eigen_support::GetThreadPoolDevice(), out_backprop_data, batches, out_backprop_height,
             out_backprop_width, out_backprop_depth, filter_data, filter_height, filter_width,
             row_dilation, col_dilation, row_stride, col_stride, padding_type, padding_top,
             padding_bottom, padding_left, padding_right, in_backprop_data, in_backprop_height,
             in_backprop_width, in_backprop_depth);
  }
};

// From tensorflow/core/kernels/conv_grad_filter_ops.cc
template <typename T> struct LaunchConv2DBackpropFilterOp
{
  void operator()(const T *out_backprop_data, int batches, int out_backprop_height,
                  int out_backprop_width, int out_backprop_depth, const T *input_data,
                  int input_height, int input_width, int input_depth, int row_dilation,
                  int col_dilation, int row_stride /* H */, int col_stride /* W */,
                  const PaddingType &padding_type, int padding_top, int padding_bottom,
                  int padding_left, int padding_right, T *filter_backprop_data,
                  int filter_backprop_height, int filter_backprop_width)
  {
    eigen_support::EigenTensor filter_backprop_t(filter_backprop_data, filter_backprop_height,
                                                 filter_backprop_width, input_depth,
                                                 out_backprop_depth);
    eigen_support::ConstEigenTensor input_t(input_data, batches, input_height, input_width,
                                            input_depth);
    eigen_support::ConstEigenTensor out_backprop_t(out_backprop_data, batches, out_backprop_height,
                                                   out_backprop_width, out_backprop_depth);

    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();

    if (padding_type != PaddingType::kNone /* EXPLICIT */)
    {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward filter will infer correct forward paddings from input tensors.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
        input_t, out_backprop_t, filter_backprop_t.dimension(1), filter_backprop_t.dimension(0),
        col_stride, row_stride, col_dilation, row_dilation);
    }
    else
    {
      // Otherwise we have to explicitly pad the input, before passing it to
      // spatial convolution backward filter.
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = {0, 0};
      paddings[1] = {padding_top, padding_bottom};
      paddings[2] = {padding_left, padding_right};
      paddings[3] = {0, 0};

      auto padded_t = input_t.pad(paddings, T(0));

      // TODO(ezhulenev): Pass explicit paddings to Eigen spatial backward
      // convolution and do not rely on tensor padding expression.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
        padded_t, out_backprop_t, filter_backprop_t.dimension(1), filter_backprop_t.dimension(0),
        col_stride, row_stride, col_dilation, row_dilation);
    }
  }
};

inline void ConvInputGrad(const ConvParams &params, const Shape &out_backprop_shape,
                          const float *out_backprop_data, const Shape &filter_shape,
                          const float *filter_data, const int padding_bottom,
                          const int padding_right, const Shape &in_backprop_shape,
                          float *in_backprop_data)
{
  const int stride_rows = params.stride_height;
  const int stride_cols = params.stride_width;
  const PaddingType padding = params.padding_type;
  const int padding_top = params.padding_values.height;
  const int padding_left = params.padding_values.width;
  assert(padding_top >= 0);
  assert(padding_bottom >= 0);
  assert(padding_left >= 0);
  assert(padding_right >= 0);
  const int dilation_rows = params.dilation_height_factor;
  const int dilation_cols = params.dilation_width_factor;

  const int batches = MatchingDim(in_backprop_shape, 0, out_backprop_shape, 0);
  const int in_backprop_depth = MatchingDim(in_backprop_shape, 3, filter_shape, 2);
  const int out_backprop_depth = MatchingDim(filter_shape, 3, out_backprop_shape, 3);
  const int in_backprop_height = in_backprop_shape.Dims(1);
  const int in_backprop_width = in_backprop_shape.Dims(2);
  const int filter_height = filter_shape.Dims(0);
  const int filter_width = filter_shape.Dims(1);
  const int out_backprop_height = out_backprop_shape.Dims(1);
  const int out_backprop_width = out_backprop_shape.Dims(2);

  if (dilation_rows != 1 || dilation_cols != 1)
    throw std::runtime_error("cker::ConvFilterGrad: not yet support dilation rates larger than 1.");

  LaunchConv2DBackpropInputOp<float>()(
    out_backprop_data, batches, out_backprop_height, out_backprop_width, out_backprop_depth,
    filter_data, filter_height, filter_width, dilation_rows, dilation_cols, stride_rows,
    stride_cols, padding, padding_top, padding_bottom, padding_left, padding_right,
    in_backprop_data, in_backprop_height, in_backprop_width, in_backprop_depth);
}

inline void ConvFilterGrad(const ConvParams &params, const Shape &out_backprop_shape,
                           const float *out_backprop_data, const Shape &input_shape,
                           const float *input_data, const int padding_bottom,
                           const int padding_right, const Shape &filter_backprop_shape,
                           float *filter_backprop_data)
{
  const int stride_rows = params.stride_height;
  const int stride_cols = params.stride_width;
  const PaddingType padding = params.padding_type;
  const int padding_top = params.padding_values.height;
  const int padding_left = params.padding_values.width;
  assert(padding_top >= 0);
  assert(padding_bottom >= 0);
  assert(padding_left >= 0);
  assert(padding_right >= 0);
  const int dilation_rows = params.dilation_height_factor;
  const int dilation_cols = params.dilation_width_factor;

  const int batches = MatchingDim(input_shape, 0, out_backprop_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_backprop_shape, 2);
  const int out_backprop_depth = MatchingDim(filter_backprop_shape, 3, out_backprop_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_backprop_height = filter_backprop_shape.Dims(0);
  const int filter_backprop_width = filter_backprop_shape.Dims(1);
  const int out_backprop_height = out_backprop_shape.Dims(1);
  const int out_backprop_width = out_backprop_shape.Dims(2);

  if (dilation_rows != 1 || dilation_cols != 1)
    throw std::runtime_error("cker::ConvFilterGrad: not yet support dilation rates larger than 1.");

  LaunchConv2DBackpropFilterOp<float>()(
    out_backprop_data, batches, out_backprop_height, out_backprop_width, out_backprop_depth,
    input_data, input_height, input_width, input_depth, dilation_rows, dilation_cols, stride_rows,
    stride_cols, padding, padding_top, padding_bottom, padding_left, padding_right,
    filter_backprop_data, filter_backprop_height, filter_backprop_width);
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_RELU_H__
