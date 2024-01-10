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

class DepthwiseConv
{
public:
  DepthwiseConv() = default;

  template <typename T> int64_t kPacketSize() const
  {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    return sizeof(Packet) / sizeof(T);
  }

  int getThreadCount() const
  {
    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();
    return d.numThreads();
  }

  template <typename T>
  void backpropInput(const DepthwiseConvParams &params, const Shape &incoming_shape,
                     const T *incoming_data, const Shape &filter_shape, const T *filter_data,
                     T *padded_filter_data, const Shape &grad_shape, T *grad_data, bool pad_filter,
                     std::vector<uint8_t *> &out_bprop, std::vector<uint8_t *> &in_bprop)
  {
    if (params.stride_height != params.stride_width)
      throw std::runtime_error("Not support different length strides");

    const int batch = MatchingDim(incoming_shape, 0, grad_shape, 0);
    const int input_depth = grad_shape.Dims(3);
    const int output_depth = incoming_shape.Dims(3);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);
    const int grad_height = grad_shape.Dims(1);
    const int grad_width = grad_shape.Dims(2);
    const int stride = params.stride_height;
    const int depth_multiplier = params.depth_multiplier;
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;

    depthwise_conv_op::LaunchDepthwiseConvBackpropInputOp<Eigen::ThreadPoolDevice, T>()(
      batch, grad_height, grad_width, input_depth, filter_height, filter_width, depth_multiplier,
      stride, pad_height, pad_width, incoming_height, incoming_width, output_depth, incoming_data,
      filter_data, padded_filter_data, grad_data, pad_filter, out_bprop, in_bprop);

    // depthwise_conv_op::DepthwiseConvBackpropInputReference<float>(
    //   batch, grad_height, grad_width, input_depth, incoming_height, incoming_width, output_depth,
    //   stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width,
    //   incoming_data, filter_data, grad_data);
  }

  template <typename T>
  void backpropFilter(const DepthwiseConvParams &params, const Shape &incoming_shape,
                      const T *incoming_data, const Shape &input_shape, const T *input_data,
                      const Shape &filter_grad_shape, T *filter_grad_data)
  {
    if (params.stride_height != params.stride_width)
      throw std::runtime_error("Not support different length strides");

    const int batch = MatchingDim(incoming_shape, 0, input_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = incoming_shape.Dims(3);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int stride = params.stride_height;
    const int depth_multiplier = params.depth_multiplier;
    const int filter_height = filter_grad_shape.Dims(1);
    const int filter_width = filter_grad_shape.Dims(2);
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;

    depthwise_conv_op::LaunchDepthwiseConvBackpropFilterOp<Eigen::ThreadPoolDevice, T>()(
      batch, input_width, input_height, input_depth, filter_width, filter_height, depth_multiplier,
      stride, pad_width, pad_height, incoming_width, incoming_height, output_depth, incoming_data,
      input_data, filter_grad_data);

    // depthwise_conv_op::DepthwiseConvBackpropFilterReference<T>(
    //   batch, input_height, input_width, input_depth, incoming_height, incoming_width,
    //   output_depth, stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width,
    //   incoming_data, input_data, filter_grad_data);
  }
};

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_DEPTHWISECONV_H__
