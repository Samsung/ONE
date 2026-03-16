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

#include "ConvolutionSpec.h"
#include "PaddingUtils.h"
#include "ShapeQuery.h"

#include <cassert>

ConvolutionSpec::ConvolutionSpec(const ::caffe::ConvolutionParameter &param) : _param(param)
{
  // NOTE Dilation is not supported, yet
  // TODO Support dilation
  assert(param.dilation().size() == 0);
}

uint32_t ConvolutionSpec::group(void) const { return _param.group(); }

uint32_t ConvolutionSpec::channel_axis(void) const
{
  return query_on(ifm_shape()).axis(axis_specifier(_param.axis()));
}

uint32_t ConvolutionSpec::pad(uint32_t spatial_axis) const
{
  assert(spatial_axis < num_spatial_axes());

  auto raw_padding = build_raw_padding().with(_param);
  auto spatial_padding = build_spatial_padding(num_spatial_axes()).with(raw_padding);

  return spatial_padding.value(spatial_axis);
}

uint32_t ConvolutionSpec::stride(uint32_t spatial_axis) const
{
  assert(spatial_axis < num_spatial_axes());

  // TODO Support stride_h/stride_w parameters
  assert(!_param.has_stride_h());
  assert(!_param.has_stride_w());

  if (_param.stride().size() == 0)
  {
    // NOTE default stride is 1
    return 1;
  }

  if (_param.stride().size() == 1)
  {
    return _param.stride(0);
  }

  assert(_param.stride().size() == num_spatial_axes());
  return _param.stride(spatial_axis);
}

uint32_t ConvolutionSpec::ker_dim(uint32_t spatial_axis) const
{
  assert(spatial_axis < num_spatial_axes());
  if (_param.kernel_size().size() == 0)
  {
    if (_param.has_kernel_h() && (spatial_axis == 0))
    {
      assert(num_spatial_axes() == 2);
      return _param.kernel_h();
    }

    if (_param.has_kernel_w() && (spatial_axis == 1))
    {
      assert(num_spatial_axes() == 2);
      return _param.kernel_w();
    }

    return 0;
  }

  assert(!_param.has_kernel_h());
  assert(!_param.has_kernel_w());
  if (_param.kernel_size().size() == 1)
  {
    return _param.kernel_size(0);
  }
  else
  {
    assert(_param.kernel_size().size() == num_spatial_axes());
    return _param.kernel_size(spatial_axis);
  }
}

nncc::core::ADT::tensor::Shape ConvolutionSpec::ker_shape(void) const
{
  nncc::core::ADT::tensor::Shape res;

  res.resize(2 + num_spatial_axes());

  res.dim(0) = ker_count();
  assert(ifm_dim(channel_axis()) % group() == 0);
  res.dim(1) = ifm_dim(channel_axis()) / group();
  for (uint32_t axis = 0; axis < num_spatial_axes(); ++axis)
  {
    res.dim(2 + axis) = ker_dim(axis);
  }

  return res;
}

nncc::core::ADT::tensor::Shape ConvolutionSpec::ofm_shape(void) const
{
  nncc::core::ADT::tensor::Shape res;

  res.resize(num_batch_axes() + 1 + num_spatial_axes());

  for (uint32_t axis = 0; axis < num_batch_axes(); ++axis)
  {
    res.dim(axis) = ifm_dim(axis);
  }

  res.dim(num_batch_axes()) = ker_count();

  for (uint32_t spatial_axis = 0; spatial_axis < num_spatial_axes(); ++spatial_axis)
  {
    const uint32_t full_axis = num_batch_axes() + 1 + spatial_axis;

    uint32_t dim = 0;

    dim += ifm_dim(full_axis) - ker_dim(spatial_axis) + 2 * pad(spatial_axis);
    dim /= stride(spatial_axis);
    dim += 1;

    res.dim(full_axis) = dim;
  }

  return res;
}
