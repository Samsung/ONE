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

#include "moco/Import/Nodes/Pad.h"

#include <moco/IR/Nodes/TFPad.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for TF Pad node
 */
class TFPadGraphUpdate final : public GraphUpdate
{
public:
  TFPadGraphUpdate(TFPad *node, std::vector<TensorName> names) : _node(node), _names(names) {}

  void input(const SymbolTable *) const override;

private:
  TFPad *_node;
  std::vector<TensorName> _names;
};

void TFPadGraphUpdate::input(const SymbolTable *table) const
{
  assert(_names.size() == 2);

  _node->input(table->node(_names[0]));
  _node->paddings(table->node(_names[1]));
}

} // namespace

namespace moco
{

bool PadGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 2)
    return false;

  return plier::tf::has_attrs(node, {"T", "Tpaddings"});
}

void PadGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect Pad node
  auto tf_pad = graph->nodes()->create<TFPad>();
  tf_pad->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_pad);

  std::vector<TensorName> add_input_names;
  add_input_names.push_back(TensorName(node.input(0))); // input
  add_input_names.push_back(TensorName(node.input(1))); // paddings

  // Queue node input update
  auto tf_pad_update = std::make_unique<TFPadGraphUpdate>(tf_pad, add_input_names);
  updates->enroll(std::move(tf_pad_update));
}

} // namespace moco
