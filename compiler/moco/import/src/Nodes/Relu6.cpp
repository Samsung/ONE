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

#include "moco/Import/Nodes/Relu6.h"

#include <moco/IR/Nodes/TFRelu6.h>

#include <memory>

namespace
{

using namespace moco;

class TFRelu6GraphUpdate final : public GraphUpdate
{
public:
  TFRelu6GraphUpdate(TFRelu6 *node, const TensorName &&name) : _node(node), _name(name) {}

  void input(const SymbolTable *) const override;

private:
  TFRelu6 *_node;
  const TensorName _name;
};

void TFRelu6GraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *target = table->node(_name);
  _node->features(target);
}

} // namespace

namespace moco
{

bool Relu6GraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  // ReLU6 node SHOULD have only one input
  if (node.input_size() != 1)
    return false;
  return true;
}

void Relu6GraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // Create a "TFRelu6" node for Relu
  auto relu_node = graph->nodes()->create<TFRelu6>();
  relu_node->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, relu_node);

  // Queue node input update
  auto update = std::make_unique<TFRelu6GraphUpdate>(relu_node, TensorName(node.input(0)));
  updates->enroll(std::move(update));
}

} // namespace moco
