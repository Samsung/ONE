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

#ifndef __LOCO_IR_FILTER_CODEC_H__
#define __LOCO_IR_FILTER_CODEC_H__

#include "loco/IR/FilterShape.h"
#include "loco/IR/FilterIndex.h"

#include "loco/IR/TensorShape.h"
#include "loco/IR/TensorIndex.h"

#include "loco/IR/CastHelpers.h"

namespace loco
{

/**
 * @brief Decribe how to build a (convolution) filter from a tensor
 *
 * Let us assume that "enc" is a filter encoder.
 *
 * Given a tensor "inp" and its shape "inp.shape", "enc" builds a filter
 * "out" as follows:
 *
 * for each valid filter index (referred to as filter_index below) for enc.shape(inp.shape)
 *   out.at(filter_index) = inp.at(enc.value(filter_index))
 */
struct FilterEncoder
{
  virtual ~FilterEncoder() = default;

  virtual FilterShape shape(const TensorShape &shape) const = 0;
  virtual TensorIndex value(const FilterIndex &index) const = 0;
};

/**
 * @brief Decribe how to build a a tensor from a filter
 */
struct FilterDecoder
{
  virtual ~FilterDecoder() = default;

  virtual TensorShape shape(const FilterShape &shape) const = 0;
  virtual FilterIndex value(const TensorIndex &index) const = 0;
};

/**
 * @brief A helper dynamic_cast that throws when failed
 */
template <typename T> T must_cast(FilterEncoder *node)
{
  return _must_cast<T, FilterEncoder *>(node);
}

template <typename T> T must_cast(const FilterEncoder *node)
{
  return _must_cast<T, const FilterEncoder *>(node);
}

// TODO add must_cast for FilterDecoder

} // namespace loco

#endif // __LOCO_IR_FILTER_CODEC_H__
