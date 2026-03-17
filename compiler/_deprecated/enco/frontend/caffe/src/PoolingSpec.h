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

#ifndef __POOLING_SPEC_H__
#define __POOLING_SPEC_H__

#include <caffe/proto/caffe.pb.h>

#include <nncc/core/ADT/tensor/Shape.h>

enum class PoolingMethod
{
  Max,
  Avg
};

class PoolingSpec
{
public:
  PoolingSpec(const ::caffe::PoolingParameter &param);

public:
  const nncc::core::ADT::tensor::Shape &ifm_shape(void) const { return _ifm_shape; }
  void ifm_shape(const nncc::core::ADT::tensor::Shape &shape) { _ifm_shape = shape; }

public:
  PoolingMethod method(void) const;

public:
  uint32_t window_height(void) const;
  uint32_t window_width(void) const;

public:
  uint32_t vertical_pad(void) const;
  uint32_t horizontal_pad(void) const;

public:
  uint32_t vertical_stride(void) const;
  uint32_t horizontal_stride(void) const;

public:
  nncc::core::ADT::tensor::Shape ofm_shape(void) const;

private:
  const ::caffe::PoolingParameter &_param;
  nncc::core::ADT::tensor::Shape _ifm_shape;
};

#endif // __POOLING_SPEC_H__
