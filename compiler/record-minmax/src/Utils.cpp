/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Utils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/DataTypeHelper.h>

namespace record_minmax
{

uint32_t numElements(const luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
  {
    if (not node->dim(i).known())
      throw std::runtime_error("Unknown dimension found in " + node->name());

    num_elements *= node->dim(i).value();
  }

  return num_elements;
}

size_t getTensorSize(const luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  uint32_t elem_size = luci::size(node->dtype());
  return numElements(node) * elem_size;
}

} // namespace record_minmax
