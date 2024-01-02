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

// #include "cker/operation/Helper/Tensor.h"
// #include "cker/eigen/eigen_backward_spatial_convolutions.h"
// #include "cker/eigen/EigenSupport.h"
// #include "cker/eigen/Utils.h"
#include "cker/eigen/depthwise_conv_op.h"
#include "cker/Shape.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{
namespace train
{

inline void DepthwiseConvInputGrad(const DepthwiseConvParams &params, const Shape &incoming_shape,
                                   const float *incoming_data, const Shape &filter_shape,
                                   const float *filter_data, const Shape &grad_shape,
                                   float *grad_data)
{
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

  // depthwise_conv_op::LaunchDepthwiseConvBackpropInputOp<Eigen::ThreadPoolDevice, float>()(
  //   batch, grad_height, grad_width, input_depth, incoming_height, incoming_width, output_depth,
  //   stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
  //   filter_data, grad_data);

  depthwise_conv_op::DepthwiseConvBackpropInputReference<float>(
    batch, grad_height, grad_width, input_depth, incoming_height, incoming_width, output_depth,
    stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
    filter_data, grad_data);
}

inline void DepthwiseConvFilterGradRef(const DepthwiseConvParams &params,
                                       const Shape &incoming_shape, const float *incoming_data,
                                       const Shape &input_shape, const float *input_data,
                                       const Shape &filter_grad_shape, float *filter_grad_data)
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

  depthwise_conv_op::DepthwiseConvBackpropFilterReference<float>(
    batch, input_height, input_width, input_depth, incoming_height, incoming_width, output_depth,
    stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
    input_data, filter_grad_data);
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_DEPTHWISECONV_H__
