/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/test/TestShape.h"

/**
 * @note This file does not hold any test cases but provides methods for tests
 */

namespace luci
{
namespace test
{

void set_shape_vector(loco::TensorShape *shape, const ShapeU32 &values)
{
  uint32_t r = 0;
  shape->rank(values.size());
  for (auto v : values)
    shape->dim(r++).set(v);
}

void set_shape_vector(luci::CircleConst *const_node, const ShapeI32 &values)
{
  const_node->rank(1);
  const_node->dim(0).set(values.size());
  const_node->shape_status(luci::ShapeStatus::VALID);
  const_node->dtype(loco::DataType::S32);
  const_node->size<loco::DataType::S32>(values.size());
  uint32_t idx = 0;
  for (auto val : values)
    const_node->at<loco::DataType::S32>(idx++) = val;
}

uint32_t num_elements(const ShapeU32 shape)
{
  uint32_t result = 1;
  for (auto val : shape)
    result = result * val;
  return result;
}

} // namespace test
} // namespace luci
