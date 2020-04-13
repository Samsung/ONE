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

#include "PaddingUtils.h"

#include <cassert>

//
// Section: Raw Padding Builder
//
RawPadding RawPaddingBuilder::with(const ::caffe::ConvolutionParameter &param) const
{
  RawPadding res;

  if (param.has_pad_h() || param.has_pad_w())
  {
    assert(param.pad().size() == 0);
    assert(param.has_pad_h() && param.has_pad_w());

    res.resize(2);
    res.value(0) = param.pad_h();
    res.value(1) = param.pad_w();
  }
  else
  {
    // NOTE pad and pad_h/pad_w cannot be specified at the same time
    //
    // Reference: BaseConvolutionLayer<Dtype>::LayerSetUp in base_conv_layer.cpp
    assert(!param.has_pad_h() && !param.has_pad_w());

    uint32_t rank = param.pad().size();

    res.resize(rank);
    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      res.value(axis) = param.pad(axis);
    }
  }

  return res;
}

RawPadding RawPaddingBuilder::with(const ::caffe::PoolingParameter &param) const
{
  RawPadding res;

  if (param.has_pad_h() || param.has_pad_w())
  {
    assert(!param.has_pad());
    assert(param.has_pad_h() && param.has_pad_w());

    res.resize(2);
    res.value(0) = param.pad_h();
    res.value(1) = param.pad_w();
  }
  else
  {
    // NOTE pad and pad_h/pad_w cannot be specified at the same time
    //
    // Reference: PoolingLayer<Dtype>::LayerSetUp in pooling_layer.cpp
    assert(!param.has_pad_h() && !param.has_pad_w());

    if (param.has_pad())
    {
      res.resize(1);
      res.value(0) = param.pad();
    }
  }

  return res;
}

RawPaddingBuilder build_raw_padding(void) { return RawPaddingBuilder{}; }

//
// Section: Spatial Padding Builder
//
SpatialPadding SpatialPaddingBuilder::with(const RawPadding &raw) const
{
  const auto spatial_rank = _spatial_rank;

  SpatialPadding res;

  res.resize(spatial_rank);

  if (raw.count() == 0)
  {
    // NOTE default padding is 0
    for (uint32_t spatial_axis = 0; spatial_axis < spatial_rank; ++spatial_axis)
    {
      res.value(spatial_axis) = 0;
    }
  }
  else if (raw.count() == 1)
  {
    // NOTE One-for-all scheme
    for (uint32_t spatial_axis = 0; spatial_axis < spatial_rank; ++spatial_axis)
    {
      res.value(spatial_axis) = raw.value(0);
    }
  }
  else
  {
    // NOTE One-to-one scheme
    assert(raw.count() == spatial_rank);
    for (uint32_t spatial_axis = 0; spatial_axis < spatial_rank; ++spatial_axis)
    {
      res.value(spatial_axis) = raw.value(spatial_axis);
    }
  }

  return res;
}

SpatialPaddingBuilder build_spatial_padding(uint32_t spatial_rank)
{
  return SpatialPaddingBuilder{spatial_rank};
}
