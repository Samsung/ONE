/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_DEPTHWISE_FILTER_CODEC_H__
#define __LOCO_IR_DEPTHWISE_FILTER_CODEC_H__

#include "loco/IR/DepthwiseFilterShape.h"
#include "loco/IR/DepthwiseFilterIndex.h"

#include "loco/IR/TensorShape.h"
#include "loco/IR/TensorIndex.h"

namespace loco
{

/**
 * @brief Describe how to build a depthwise convolution filter from a tensor
 *
 * Let us assume that "enc" is a depthwise filter encoder.
 *
 * Given a tensor "inp" and its shape "inp.shape", "enc" builds a depthwise filter
 * "out" as follows:
 *
 * for each valid filter_index for enc.shape(inp.shape)
 *   out.at(filter_index) = inp.at(enc.value(filter_index))
 */
struct DepthwiseFilterEncoder
{
  virtual ~DepthwiseFilterEncoder() = default;

  virtual DepthwiseFilterShape shape(const TensorShape &shape) const = 0;
  virtual TensorIndex value(const DepthwiseFilterIndex &index) const = 0;
};

/**
 * @brief Describe how to build a tensor from a depthwise convolution filter
 *
 * Let us assume that "dec" is a depthwise filter decoder.
 *
 * Given a depthwise filter "inp" and its shape "inp.shape", "dec" builds a tensor
 * "out" as follows:
 *
 * for each valid tensor_index for dec.shape(inp.shape)
 *   out.at(tensor_index) = inp.at(dec.value(tensor_index))
 */
struct DepthwiseFilterDecoder
{
  virtual ~DepthwiseFilterDecoder() = default;

  virtual TensorShape shape(const DepthwiseFilterShape &shape) const = 0;
  virtual DepthwiseFilterIndex value(const TensorIndex &index) const = 0;
};

} // namespace loco

#endif // __LOCO_IR_DEPTHWISE_FILTER_CODEC_H__
