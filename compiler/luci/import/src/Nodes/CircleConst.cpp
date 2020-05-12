/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleConst.h"

#include <luci/IR/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <loco.h>
#include <oops/UserExn.h>

#include <cassert>

namespace luci
{

template <loco::DataType DT>
static void copy_data(const std::vector<uint8_t> &raw_data, uint32_t num_elements,
                      CircleConst *const_node)
{
  using T = typename loco::DataTypeImpl<DT>::Type;

  assert(raw_data.size() == num_elements * sizeof(T));
  const auto *data = reinterpret_cast<const T *>(raw_data.data());

  const_node->size<DT>(num_elements);
  for (uint32_t i = 0; i < num_elements; ++i)
  {
    const_node->at<DT>(i) = data[i];
  }
}

//
// circleconst_from_tensor() ?
//
CircleConst *create_circleconst(GraphBuilderContext *context, int32_t tensor_index)
{
  LOGGER(l);

  auto graph = context->graph();
  auto reader = context->reader();
  const auto &tensors = reader->tensors();

  // (1) create CircleConst
  auto const_node = graph->nodes()->create<CircleConst>();
  const circle::TensorT &const_tensor = *tensors[tensor_index];
  copy_tensor_attributes(const_tensor, const_node);

  INFO(l) << "[luci] NodeFinder const_node(" << tensor_index << ") -> " << const_node << std::endl;

  // (2) get number of elements
  std::vector<int32_t> const_dims = const_tensor.shape; // in NHWC
  uint32_t num_elements = 1;
  for (uint32_t r = 0; r < const_dims.size(); ++r)
  {
    num_elements = num_elements * const_dims[r];
  }

  // (3) constant values from circle buffer
  const std::vector<uint8_t> &buffer = reader->buffers()[const_tensor.buffer]->data;
  if (buffer.empty())
    throw oops::UserExn("Empty buffer");

  switch (luci_datatype(const_tensor.type))
  {
    case loco::DataType::FLOAT32:
      copy_data<loco::DataType::FLOAT32>(buffer, num_elements, const_node);
      break;

    case loco::DataType::U8:
      copy_data<loco::DataType::U8>(buffer, num_elements, const_node);
      break;

    case loco::DataType::S32:
      copy_data<loco::DataType::S32>(buffer, num_elements, const_node);
      break;

    case loco::DataType::S64:
      copy_data<loco::DataType::S64>(buffer, num_elements, const_node);
      break;

    case loco::DataType::BOOL:
      copy_data<loco::DataType::BOOL>(buffer, num_elements, const_node);
      break;

    default:
      throw oops::UserExn("Unsupported tensor type", circle::EnumNameTensorType(const_tensor.type));
  }

  return const_node;
}

} // namespace luci
