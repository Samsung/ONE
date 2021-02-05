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

#include "Utilities.h"
#include "luci/IR/Nodes/CircleConst.h"

#include <cassert>

namespace luci_codegen
{

// TODO do something with luci to simplify this shame
size_t const_node_size(const luci::CircleNode *node)
{
  assert(node->opcode() == luci::CircleOpcode::CIRCLECONST);
  auto const_node = static_cast<const luci::CircleConst *>(node);
  switch (node->dtype())
  {
    case loco::DataType::S32:
      return sizeof(std::int32_t) * const_node->size<loco::DataType::S32>();
    case loco::DataType::S64:
      return sizeof(std::int32_t) * const_node->size<loco::DataType::S64>();
    case loco::DataType::FLOAT32:
      return sizeof(std::int32_t) * const_node->size<loco::DataType::FLOAT32>();
//    case loco::DataType::FLOAT64:
//      return sizeof(std::int32_t) * const_node->size<loco::DataType::FLOAT64>(); // double is not supported in luci
  }
  return 0;
}

Halide::Type halide_type(loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      return Halide::Type(Halide::Type::Float, 32, 1);
    case loco::DataType::FLOAT64:
      return Halide::Type(Halide::Type::Float, 64, 1);
    case loco::DataType::S32:
      return Halide::Type(Halide::Type::Int, 32, 1);
    case loco::DataType::S64:
      return Halide::Type(Halide::Type::Int, 64, 1);
    default:
      assert("NYI");
  }
  return Halide::Type();
}

} // namespace luci_codegen
