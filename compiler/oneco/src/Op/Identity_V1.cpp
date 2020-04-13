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

#include "Identity.h"

#include <cassert>

namespace moco
{
namespace onnx
{

bool Identity_V1::validate(const ::onnx::NodeProto &) const { return true; }

void Identity_V1::build(const ::onnx::NodeProto &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *nodes = context->nodes();
  SymbolTable *input_names = context->input_names();

  // Create a "Forward" node for Identity
  auto forward_node = graph->nodes()->create<loco::Forward>();

  nodes->enroll(node.name(), forward_node);
  nodes->enroll(node.output(0), forward_node);

  // Record all inputs to forward_node
  for (int i = 0; i < node.input_size(); ++i)
  {
    const auto &input_name = node.input(i);
    input_names->list(forward_node, input_name);
  }
}

} // namespace onnx
} // namespace moco
