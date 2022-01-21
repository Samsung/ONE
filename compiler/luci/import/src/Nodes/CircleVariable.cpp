/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleVariable.h"

#include <luci/IR/Nodes/CircleVariable.h>
#include <luci/Log.h>

#include <cassert>
#include <ostream>
#include <string>
#include <vector>

namespace
{

std::ostream &operator<<(std::ostream &os, const luci::VectorWrapper<int32_t> &vect)
{
  uint32_t seq = 0;
  for (const auto &v : vect)
  {
    if (seq)
      os << ", ";
    os << v;
    seq++;
  }
  return os;
}

} // namespace

namespace luci
{

CircleVariable *create_circlevariable(GraphBuilderContext *context, int32_t tensor_index)
{
  LOGGER(l);

  auto graph = context->graph();
  auto reader = context->reader();
  const auto tensors = reader->tensors();
  const auto variable_tensor = tensors[tensor_index];
  assert(variable_tensor != nullptr);

  if (not variable_tensor->is_variable())
  {
    // not a variable
    return nullptr;
  }
  {
    // check if there is no buffer as we don't support this for now
    // TODO use buffer when this is enabled in Kernel
    assert(reader->buffers()[variable_tensor->buffer()] != nullptr);
    assert(reader->buffers()[variable_tensor->buffer()]->data() == nullptr);
  }

  auto variable_node = graph->nodes()->create<CircleVariable>();
  copy_tensor_attributes(variable_tensor, variable_node);
  variable_node->shape_status(luci::ShapeStatus::VALID);

  INFO(l) << "[luci] NodeFinder variable node(" << tensor_index << ") -> " << variable_node << " "
          << wrap(variable_tensor->shape()) << std::endl;

  return variable_node;
}

} // namespace luci
