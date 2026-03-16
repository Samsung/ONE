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

#ifndef __CONVOLUTION_SPEC_H__
#define __CONVOLUTION_SPEC_H__

#include <caffe/proto/caffe.pb.h>

#include <nncc/core/ADT/tensor/Shape.h>

class ConvolutionSpec
{
public:
  ConvolutionSpec(const ::caffe::ConvolutionParameter &param);

public:
  uint32_t ifm_rank(void) const { return _ifm_shape.rank(); }
  uint32_t ifm_dim(uint32_t axis) const { return _ifm_shape.dim(axis); }

  uint32_t group(void) const;

  uint32_t channel_axis(void) const;

  uint32_t num_batch_axes(void) const { return channel_axis(); }
  uint32_t num_spatial_axes(void) const { return ifm_rank() - channel_axis() - 1; }

  uint32_t pad(uint32_t spatial_axis) const;
  uint32_t stride(uint32_t spatial_axis) const;
  uint32_t ker_dim(uint32_t spatial_axis) const;

public:
  const nncc::core::ADT::tensor::Shape &ifm_shape(void) const { return _ifm_shape; }
  void ifm_shape(const nncc::core::ADT::tensor::Shape &shape) { _ifm_shape = shape; }

public:
  uint32_t ker_count(void) const { return _param.num_output(); }
  nncc::core::ADT::tensor::Shape ker_shape(void) const;

public:
  nncc::core::ADT::tensor::Shape ofm_shape(void) const;

private:
  const ::caffe::ConvolutionParameter &_param;
  nncc::core::ADT::tensor::Shape _ifm_shape;
};
#endif // __CONVOLUTION_SPEC_H__
