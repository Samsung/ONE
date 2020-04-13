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
#include <stdex/Memory.h>
#include <oops/UserExn.h>

#include <cassert>

namespace luci
{

//
// circleconst_from_tensor() ?
//
CircleConst *create_circleconst(GraphBuilderContext *context, int32_t tensor_index)
{
  LOGGER(l);

  auto graph = context->graph();
  auto reader = context->reader();
  auto opfinder = context->opfinder();
  auto tensorfinder = context->tensorfinder();
  auto nodefinder = context->nodefinder();
  auto tensors = reader->tensors();

  // (1) create CircleConst
  auto const_node = graph->nodes()->create<CircleConst>();
  auto const_tensor = tensors->Get(tensor_index);
  opfinder->enroll(const_node, nullptr);
  tensorfinder->enroll(const_node, const_tensor);
  nodefinder->enroll(tensor_index, const_node);

  INFO(l) << "[luci] NodeFinder const_node(" << tensor_index << ") -> " << const_node << std::endl;

  // (2) set data_type to CircleConst
  const_node->dtype(luci_datatype(const_tensor));

  // (3) set shape to CicleConst
  assert(const_tensor->shape());
  std::vector<int32_t> const_dims = as_index_vector(const_tensor->shape()); // in NHWC
  const_node->rank(const_dims.size());
  uint32_t num_elements = 1;
  for (uint32_t r = 0; r < const_dims.size(); ++r)
  {
    const_node->dim(r) = loco::Dimension(const_dims[r]);
    num_elements = num_elements * const_dims[r];
  }

  // (4) constant values from circle buffer
  uint32_t const_buff_idx = const_tensor->buffer();
  const uint8_t *const_buff_data = nullptr;
  size_t const_buff_size = reader->buffer_info(const_buff_idx, &const_buff_data);
  switch (luci_datatype(const_tensor))
  {
    case loco::DataType::FLOAT32:
    {
      // NOTE assert(const_buff_size == num_elements * sizeof(float)) will drop
      // unused variables compilation error in release build.
      if (const_buff_size != num_elements * sizeof(float))
        throw oops::UserExn("Invalid Buffer size", "FLOAT32");
      const float *float_cb = reinterpret_cast<const float *>(const_buff_data);
      const_node->size<loco::DataType::FLOAT32>(num_elements);
      for (uint32_t ele = 0; ele < num_elements; ++ele)
        const_node->at<loco::DataType::FLOAT32>(ele) = float_cb[ele];
      break;
    }

    case loco::DataType::U8:
    {
      if (const_buff_size != num_elements * sizeof(uint8_t))
        throw oops::UserExn("Invalid Buffer size", "UINT8");
      const uint8_t *uint8_cb = reinterpret_cast<const uint8_t *>(const_buff_data);
      const_node->size<loco::DataType::U8>(num_elements);
      for (uint32_t ele = 0; ele < num_elements; ++ele)
        const_node->at<loco::DataType::U8>(ele) = uint8_cb[ele];
      break;
    }

    case loco::DataType::S32:
    {
      if (const_buff_size != num_elements * sizeof(int32_t))
        throw oops::UserExn("Invalid Buffer size", "INT32");
      const int32_t *int32_cb = reinterpret_cast<const int32_t *>(const_buff_data);
      const_node->size<loco::DataType::S32>(num_elements);
      for (uint32_t ele = 0; ele < num_elements; ++ele)
        const_node->at<loco::DataType::S32>(ele) = int32_cb[ele];
      break;
    }

    default:
      assert(false);
  }

  return const_node;
}

} // namespace luci
