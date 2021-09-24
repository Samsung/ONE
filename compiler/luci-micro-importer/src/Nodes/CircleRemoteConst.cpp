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

#include "luci/Import/Nodes/CircleRemoteConst.h"

#include <luci/Log.h>

#include <ostream>
#include <string>
#include <vector>

namespace
{

// helper for node shape logging
std::ostream &operator<<(std::ostream &os, luci::VectorWrapper<int32_t> vect)
{
  uint32_t seq = 0;
  for (auto const &v : vect)
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

CircleRemoteConst *create_circle_remote_const(GraphBuilderContext *context, int32_t tensor_index)
{
  LOGGER(l);

  auto graph = context->graph();
  auto reader = context->reader();
  const auto tensors = reader->tensors();
  auto const const_tensor = tensors[tensor_index];
  assert(const_tensor != nullptr);

  auto const buffer = wrap(reader->buffers()[const_tensor->buffer()]->data());
  auto const const_dims = wrap(const_tensor->shape()); // in NHWC
  if (const_dims.empty() && buffer.empty())
  {
    // unknown shape tensor and scalar tensor
    return nullptr;
  }

  // if tensor_index is used as output to some other operator, this is not a constant
  auto tensoroutputs = context->tensoroutputs();
  if (tensoroutputs->find(tensor_index))
  {
    // other operator output tensor
    return nullptr;
  }

  uint32_t num_elements = 1;
  for (uint32_t r = 0; r < const_dims.size(); ++r)
  {
    num_elements = num_elements * const_dims[r];
  }

  if (buffer.empty() && num_elements > 0)
  {
    // normal empty tensor
    return nullptr;
  }

  auto const_node = graph->nodes()->create<CircleRemoteConst>();
  copy_tensor_attributes(const_tensor, const_node);
  const_node->shape_status(luci::ShapeStatus::VALID);
  INFO(l) << "[luci] NodeFinder const_node(" << tensor_index << ") -> " << const_node << " "
          << const_dims << std::endl;
  if (num_elements > 0)
  {
    const_node->bind_buffer(buffer.data(), buffer.size());
  }

  return const_node;
}

} // namespace luci
