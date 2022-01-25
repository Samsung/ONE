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

#include "CircleReferencingConst.h"

#include <vector>

namespace
{

// helper struct which describes data loaded to custom_options of CircleReferencingConst node
struct ConstDataReference
{
  const uint8_t *data = nullptr;
  uint32_t size = 0;
};

} // namespace

namespace luci_interpreter
{
using namespace luci;

CircleNode *CircleReferencingConstNodeBuilder::build(TensorIndex tensor_index,
                                                     GraphBuilderContext *context) const
{
  assert(tensor_index >= 0);

  const auto graph = context->graph();
  const auto reader = context->reader();
  const auto tensors = reader->tensors();
  auto const const_tensor = tensors[tensor_index];
  assert(const_tensor != nullptr);
  if (const_tensor->is_variable())
  {
    // Create CircleVariable for variable
    return nullptr;
  }

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

  // create CircleReferencingConst
  auto custom_node = graph->nodes()->create<CircleCustom>(0, 1);
  {
    custom_node->custom_code("CircleReferencingConst");

    copy_tensor_attributes(const_tensor, custom_node);
    custom_node->shape_status(luci::ShapeStatus::VALID);

    // custom options stores size of buffer and pointer's value to buffer's data
    {
      std::vector<uint8_t> custom_options(sizeof(ConstDataReference));
      {
        auto &const_data_ref = *reinterpret_cast<ConstDataReference *>(custom_options.data());
        const_data_ref = {buffer.data(), buffer.size()};
      }
      custom_node->custom_options(custom_options);
    }
  }

  // Output of CircleCustom node presented with CircleConstNode
  auto out_node = graph->nodes()->create<CircleCustomOut>();
  {
    out_node->index(0);
    out_node->input(custom_node);

    copy_tensor_attributes(const_tensor, out_node);
    out_node->shape_status(luci::ShapeStatus::VALID);
  }

  return out_node;
}

} // namespace luci_interpreter
