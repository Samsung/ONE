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

#include "moco/Import/Nodes/Reshape.h"

#include <moco/IR/Nodes/TFReshape.h>

#include <moco/Names.h>
#include <plier/tf/Convert.h>
#include <loco.h>

#include <memory>
#include <cassert>
#include <stdexcept>

namespace
{
using namespace moco;

class ReshapeGraphUpdate final : public GraphUpdate
{
public:
  ReshapeGraphUpdate(TFReshape *node, std::vector<TensorName> names) : _node(node), _names(names) {}

  void input(const SymbolTable *) const override;

private:
  TFReshape *_node;
  std::vector<TensorName> _names;
};

void ReshapeGraphUpdate::input(const SymbolTable *node_table) const
{
  assert(_names.size() == 2);

  auto tensor_node = node_table->node(_names[0]);
  auto shape_node = node_table->node(_names[1]);

  assert(tensor_node != nullptr);
  assert(shape_node != nullptr);

  _node->tensor(tensor_node);
  _node->shape(shape_node);
}

} // namespace

namespace moco
{

bool ReshapeGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  // Tensorflow Reshape has 2 inputs: tensor & shape
  if (node.input_size() != 2)
    return false;

  // TODO Assert Tshape value is DT_INT32?
  return plier::tf::has_attrs(node, {"T", "Tshape"});
}

void ReshapeGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // name of loco nodes
  std::string reshape_name = node.name();

  auto reshape = graph->nodes()->create<TFReshape>();
  reshape->name(node.name());

  // save the name for graph link updates
  TensorName output_name(reshape_name, 0);
  tensor_names->enroll(output_name, reshape);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // tensor
  input_names.push_back(TensorName(node.input(1))); // shape

  // Queue node input update
  auto update = std::make_unique<ReshapeGraphUpdate>(reshape, input_names);

  updates->enroll(std::move(update));
}

} // namespace moco
