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

#ifndef __LOCO_IR_FEATURE_CODEC_H__
#define __LOCO_IR_FEATURE_CODEC_H__

#include "loco/IR/FeatureShape.h"
#include "loco/IR/FeatureIndex.h"

#include "loco/IR/TensorShape.h"
#include "loco/IR/TensorIndex.h"

#include "loco/IR/CastHelpers.h"

#include <memory>

namespace loco
{

/**
 * @brief Decribe how to build a (convolution) feature map from a tensor
 *
 * Let us assume that "enc" is a feature encoder.
 *
 * Given a tensor "inp" and its shape "inp.shape", "enc" builds a feature map
 * "out" as follows:
 *
 * for each valid feature index (referred to as feature_idx below) for enc.shape(inp.shape)
 *   out.at(feature_index) = inp.at(enc.value(feature_index))
 */
struct FeatureEncoder
{
  virtual ~FeatureEncoder() = default;

  virtual FeatureShape shape(const TensorShape &shape) const = 0;
  virtual TensorIndex value(const FeatureIndex &index) const = 0;

  virtual std::unique_ptr<FeatureEncoder> clone(void) const = 0;
};

/**
 * @brief Describe how to build a tensor from a (convolution) feature map
 *
 * Let us assume that "dec" is a feature decoder.
 *
 * Given a feature map "inp" and its shape "inp.shape", "dec" builds a tensor
 * "out" as follows:
 *
 * for each valid tensor index (referred to as tensor_index below) for dec.shape(inp.shape)
 *   out.at(tensor_index) = inp.at(dec.value(tensor_index))
 *
 * NOTE "inp" is a feature value and "out" is a tensor value in this example.
 */
struct FeatureDecoder
{
  virtual ~FeatureDecoder() = default;

  virtual TensorShape shape(const FeatureShape &) const = 0;
  virtual FeatureIndex value(const TensorIndex &) const = 0;

  virtual std::unique_ptr<FeatureDecoder> clone(void) const = 0;
};

/**
 * @brief A helper dynamic_cast that throws when failed
 */
template <typename T> T must_cast(FeatureEncoder *node)
{
  return _must_cast<T, FeatureEncoder *>(node);
}

template <typename T> T must_cast(const FeatureEncoder *node)
{
  return _must_cast<T, const FeatureEncoder *>(node);
}

template <typename T> T must_cast(FeatureDecoder *node)
{
  return _must_cast<T, FeatureDecoder *>(node);
}

template <typename T> T must_cast(const FeatureDecoder *node)
{
  return _must_cast<T, const FeatureDecoder *>(node);
}

} // namespace loco

#endif // __LOCO_IR_FEATURE_CODEC_H__
