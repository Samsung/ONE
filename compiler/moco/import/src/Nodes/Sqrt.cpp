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

#include "moco/Import/Nodes/Sqrt.h"

#include <moco/IR/Nodes/TFSqrt.h>

#include <loco.h>

#include <memory>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for TF Sqrt node
 */
class TFSqrtGraphUpdate final : public GraphUpdate
{
public:
  TFSqrtGraphUpdate(TFSqrt *node, TensorName &&name) : _node(node), _name(name) {}

  void input(const SymbolTable *) const override;

private:
  TFSqrt *_node;
  TensorName _name;
};

void TFSqrtGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *target = table->node(_name);
  _node->x(target);
}

} // namespace

namespace moco
{

bool SqrtGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  return node.input_size() == 1;
}

void SqrtGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect Sqrt node
  auto tf_sqrt = graph->nodes()->create<TFSqrt>();
  tf_sqrt->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_sqrt);

  // Queue node input update
  auto tf_sqrt_update = std::make_unique<TFSqrtGraphUpdate>(tf_sqrt, TensorName(node.input(0)));
  updates->enroll(std::move(tf_sqrt_update));
}

} // namespace moco
