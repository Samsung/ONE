/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PoolingSpec.h"
#include "PaddingUtils.h"

#include <map>
#include <cassert>

PoolingSpec::PoolingSpec(const ::caffe::PoolingParameter &param) : _param(param)
{
  // DO NOTHING
}

PoolingMethod PoolingSpec::method(void) const
{
  if (!_param.has_pool())
  {
    // Default pooling method is MAX
    // Reference: http://caffe.berkeleyvision.org/tutorial/layers/pooling.html
    return PoolingMethod::Max;
  }

  std::map<::caffe::PoolingParameter_PoolMethod, PoolingMethod> methods;

  // NOTE STOCHASTIC Pooling is not supported, yet
  // TODO Support STOCHASTIC Pooling
  methods[::caffe::PoolingParameter_PoolMethod_MAX] = PoolingMethod::Max;
  methods[::caffe::PoolingParameter_PoolMethod_AVE] = PoolingMethod::Avg;

  assert(_param.has_pool());
  return methods.at(_param.pool());
}

uint32_t PoolingSpec::window_height(void) const
{
  // NOTE Global pooling is not supported, yet
  // TODO Support global pooling
  assert(!_param.global_pooling());

  if (_param.has_kernel_h())
  {
    return _param.kernel_h();
  }

  assert(_param.has_kernel_size());
  return _param.kernel_size();
}

uint32_t PoolingSpec::window_width(void) const
{
  // NOTE Global pooling is not supported, yet
  // TODO Support global pooling
  assert(!_param.global_pooling());

  if (_param.has_kernel_w())
  {
    return _param.kernel_w();
  }

  assert(_param.has_kernel_size());
  return _param.kernel_size();
}

uint32_t PoolingSpec::vertical_pad(void) const
{
  // NOTE The input of Pooling SHOULD BE a rank-4 tensor.
  // Reference: PoolingLayer<Dtype>::Reshape in pooling_layer.cpp
  auto raw_padding = build_raw_padding().with(_param);
  auto spatial_padding = build_spatial_padding(2 /* SPATIAL RANK */).with(raw_padding);
  return spatial_padding.value(0 /* H */);
}

uint32_t PoolingSpec::horizontal_pad(void) const
{
  // NOTE The input of Pooling SHOULD BE a rank-4 tensor.
  // Reference: PoolingLayer<Dtype>::Reshape in pooling_layer.cpp
  auto raw_padding = build_raw_padding().with(_param);
  auto spatial_padding = build_spatial_padding(2 /* SPATIAL RANK */).with(raw_padding);
  return spatial_padding.value(1 /* W */);
}

uint32_t PoolingSpec::vertical_stride(void) const
{
  if (_param.has_stride_h())
  {
    return _param.stride_h();
  }

  if (_param.has_stride())
  {
    return _param.stride();
  }

  return 1;
}

uint32_t PoolingSpec::horizontal_stride(void) const
{
  if (_param.has_stride_w())
  {
    return _param.stride_w();
  }

  if (_param.has_stride())
  {
    return _param.stride();
  }

  return 1;
}

nncc::core::ADT::tensor::Shape PoolingSpec::ofm_shape(void) const
{
  nncc::core::ADT::tensor::Shape res;

  // NOTE Caffe supports only pooling over rank-4 tensor
  assert(_ifm_shape.rank() == 4);
  res.resize(4);

  // N (= the number of bacths) SHOULD be same
  res.dim(0) = _ifm_shape.dim(0);
  // C (= the number of chaanels) SHOULD be same
  res.dim(1) = _ifm_shape.dim(1);

  // H and W are derived from IFM, Window, and Padding
  const auto effective_input_height = _ifm_shape.dim(2) + 2 * vertical_pad() - window_height();
  const auto effective_input_width = _ifm_shape.dim(3) + 2 * horizontal_pad() - window_width();
  // TODO Remove the following asserts
  assert(effective_input_height % vertical_stride() == 0);
  assert(effective_input_width % horizontal_stride() == 0);
  res.dim(2) = effective_input_height / vertical_stride() + 1;
  res.dim(3) = effective_input_width / horizontal_stride() + 1;
  return res;
}
