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

#ifndef __LOCO_IR_TENSOR_SHAPE_H__
#define __LOCO_IR_TENSOR_SHAPE_H__

#include "loco/IR/Dimension.h"

#include <initializer_list>
#include <vector>

namespace loco
{

class TensorShape
{
public:
  TensorShape() = default;
  TensorShape(std::initializer_list<Dimension> dims) : _dims(dims.begin(), dims.end()) {}

public:
  uint32_t rank(void) const { return _dims.size(); }
  void rank(uint32_t r) { _dims.resize(r); }

  const Dimension &dim(uint32_t axis) const { return _dims.at(axis); }
  Dimension &dim(uint32_t axis) { return _dims.at(axis); }

private:
  std::vector<Dimension> _dims;
};

/**
 * @brief Return the number of elements in a tensor of given shape
 *
 * NOTE 1.
 *
 *  "volume" returns 1 if the rank is 0.
 *
 * NOTE 2.
 *
 *  "caller" SHOULD pass a valid shape that has no unknown dimension.
 *  - The behavior of "volume" on invalid is undefined.
 *
 */
uint32_t element_count(const loco::TensorShape *tensor_shape);

/**
 * @brief '==' operator for TensorShape
 */
bool operator==(const TensorShape &lhs, const TensorShape &rhs);

} // namespace loco

#endif // __LOCO_IR_TENSOR_SHAPE_H__
