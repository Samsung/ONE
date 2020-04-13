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

#include "Constant.h"
#include "Convert.h"

#include <cassert>

namespace moco
{
namespace onnx
{

bool Constant_V9::validate(const ::onnx::NodeProto &node) const
{
  if (node.attribute_size() == 0 || !node.attribute(0).has_t())
    return false;

  return true;
}

void Constant_V9::build(const ::onnx::NodeProto &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *nodes = context->nodes();

  // Create a "ConstGen" node for Constant
  auto const_node = graph->nodes()->create<loco::ConstGen>();
  auto tensor_attribute = node.attribute().Get(0).t();
  const_node->dtype(as_loco_datatype(tensor_attribute.data_type()));
  const_node->rank(tensor_attribute.dims_size());
  // TODO Support other data types
  assert(const_node->dtype() == loco::DataType::FLOAT32);
  const_node->size<loco::DataType::FLOAT32>(tensor_attribute.float_data_size());

  for (uint32_t i = 0; i < const_node->rank(); ++i)
  {
    const_node->dim(i) = tensor_attribute.dims(i);
  }

  // TODO Support other data types
  for (int i = 0; i < tensor_attribute.float_data_size(); ++i)
  {
    const_node->at<loco::DataType::FLOAT32>(i) = tensor_attribute.float_data(i);
  }

  nodes->enroll(node.name(), const_node);
  nodes->enroll(node.output(0), const_node);
}

} // namespace onnx
} // namespace moco
