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

#include "moco/Import/Nodes/Identity.h"

#include <moco/IR/Nodes/TFIdentity.h>

#include <moco/Names.h>
#include <loco.h>

#include <memory>
#include <vector>

namespace
{

using namespace moco;

class TFIdentityGraphUpdate final : public GraphUpdate
{
public:
  TFIdentityGraphUpdate(TFIdentity *node, const std::vector<TensorName> &names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFIdentity *_node;
  const std::vector<TensorName> _names;
};

void TFIdentityGraphUpdate::input(const SymbolTable *tensor_names) const
{
  for (auto &name : _names)
  {
    loco::Node *target = tensor_names->node(name);
    _node->input(target);
  }
}

} // namespace

namespace moco
{

bool IdentityGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() < 1) // from TensorFlow lite toco
    return false;

  return true;
}

void IdentityGraphBuilder::build(const tensorflow::NodeDef &node,
                                 GraphBuilderContext *context) const
{
  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // Create a Identity node
  auto identity_node = graph->nodes()->create<TFIdentity>();
  identity_node->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, identity_node);

  // Queue node input update
  // TODO: Check if we really need multiple input handlings
  std::vector<TensorName> names;
  for (int i = 0; i < node.input_size(); ++i)
  {
    names.emplace_back(TensorName(node.input(i)));
  }
  auto update = std::make_unique<TFIdentityGraphUpdate>(identity_node, names);
  updates->enroll(std::move(update));
}

} // namespace moco
