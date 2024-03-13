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

using namespace luci;

template <loco::DataType DT>
void copy_data(const VectorWrapper<uint8_t> &raw_data, uint32_t num_elements,
               CircleConst *const_node)
{
  using T = typename loco::DataTypeImpl<DT>::Type;

  // TODO calculate the exact buffer size of sparse tensor
  if (const_node->sparsityparam())
  {
    num_elements = raw_data.size() / sizeof(T);
  }

  assert(raw_data.size() == num_elements * sizeof(T));
  const auto *data = reinterpret_cast<const T *>(raw_data.data());

  const_node->size<DT>(num_elements);
  for (uint32_t i = 0; i < num_elements; ++i)
  {
    const_node->at<DT>(i) = data[i];
  }
}

template <>
void copy_data<loco::DataType::STRING>(const VectorWrapper<uint8_t> &raw_data,
                                       uint32_t num_elements, CircleConst *const_node)
{
  assert(const_node->sparsityparam() == nullptr);

  const auto *data = reinterpret_cast<const char *>(raw_data.data());
  const auto *i32d = reinterpret_cast<const int32_t *>(raw_data.data());

  // de-serialize string data
  //   int32_t count
  //   int32_t offsets[count + 1]
  //   string  values[count]
  assert(static_cast<uint32_t>(*i32d) == num_elements);
  i32d++; // skip count

  std::vector<int32_t> offsets;
  offsets.push_back(*i32d++);
  for (uint32_t i = 0; i < num_elements; ++i)
  {
    offsets.push_back(*i32d++);
  }
  assert(offsets.size() == num_elements + 1);

  const_node->size<loco::DataType::STRING>(num_elements);
  for (uint32_t i = 0; i < num_elements; ++i)
  {
    int32_t start = offsets[i];
    int32_t next = offsets[i + 1];

    std::string value(data + start, next - start);
    const_node->at<loco::DataType::STRING>(i) = value;
  }
}

} // namespace

namespace luci
{

CircleNode *CircleConstNodeBuilder::build(TensorIndex tensor_index,
                                          GraphBuilderContext *context) const
{
  assert(tensor_index >= 0);
  LOGGER(l);

  auto graph = context->graph();
  auto reader = context->reader();
  const auto tensors = reader->tensors();
  const auto const_tensor = tensors[tensor_index];
  assert(const_tensor != nullptr);
  if (const_tensor->is_variable())
  {
    // Create CircleVariable for variable
    //return nullptr;
  }

  assert(reader->buffers()[const_tensor->buffer()] != nullptr);
  const auto buffer = wrap(reader->buffers()[const_tensor->buffer()]->data());
  const auto const_dims = wrap(const_tensor->shape()); // in NHWC
  if (const_dims.size() == 0 && buffer.empty())
  {
    // unknown shape tensor and scalar tensor
    //return nullptr;
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
    //return nullptr;
  }

  auto const_node = graph->nodes()->create<CircleConst>();
  copy_tensor_attributes(const_tensor, const_node);
  const_node->shape_status(luci::ShapeStatus::VALID);
  INFO(l) << "[luci] NodeFinder const_node(" << tensor_index << ") -> " << const_node << " "
          << const_dims << std::endl;

  if (buffer.empty())
    return const_node;
  if (num_elements > 0)
  {
    switch (luci_datatype(const_tensor->type()))
    {
      case loco::DataType::FLOAT32:
        copy_data<loco::DataType::FLOAT32>(buffer, num_elements, const_node);
        break;

      case loco::DataType::FLOAT16:
        copy_data<loco::DataType::FLOAT16>(buffer, num_elements, const_node);
        break;

      case loco::DataType::U8:
        copy_data<loco::DataType::U8>(buffer, num_elements, const_node);
        break;

      case loco::DataType::S8:
        copy_data<loco::DataType::S8>(buffer, num_elements, const_node);
        break;

      case loco::DataType::S16:
        copy_data<loco::DataType::S16>(buffer, num_elements, const_node);
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

      case loco::DataType::STRING:
        copy_data<loco::DataType::STRING>(buffer, num_elements, const_node);
        break;

      default:
        throw oops::UserExn("Unsupported tensor type",
                            circle::EnumNameTensorType(const_tensor->type()));
    }
  }

  return const_node;
}

} // namespace luci
