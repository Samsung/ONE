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

#include "moco/Import/Nodes/SquaredDifference.h"

#include <moco/IR/Nodes/TFSquaredDifference.h>

#include <loco.h>
#include <stdex/Memory.h>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for TF SquaredDifference node
 */
class TFSquaredDifferenceGraphUpdate final : public GraphUpdate
{
public:
  TFSquaredDifferenceGraphUpdate(TFSquaredDifference *node, std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFSquaredDifference *_node;
  std::vector<TensorName> _names;
};

void TFSquaredDifferenceGraphUpdate::input(const SymbolTable *table) const
{
  assert(_names.size() == 2);

  _node->x(table->node(_names[0]));
  _node->y(table->node(_names[1]));
}

} // namespace

namespace moco
{

bool SquaredDifferenceGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  return node.input_size() == 2;
}

void SquaredDifferenceGraphBuilder::build(const tensorflow::NodeDef &node,
                                          GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect SquaredDifference node
  auto tf_sqdiff = graph->nodes()->create<TFSquaredDifference>();
  tf_sqdiff->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_sqdiff);

  std::vector<TensorName> add_input_names;
  add_input_names.push_back(TensorName(node.input(0))); // x
  add_input_names.push_back(TensorName(node.input(1))); // y

  // Queue node input update
  auto tf_sqrt_update =
    stdex::make_unique<TFSquaredDifferenceGraphUpdate>(tf_sqdiff, add_input_names);
  updates->enroll(std::move(tf_sqrt_update));
}

} // namespace moco
