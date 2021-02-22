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

#include "moco/Import/Nodes/FakeQuantWithMinMaxVars.h"

#include <moco/IR/Nodes/TFFakeQuantWithMinMaxVars.h>

#include <moco/Names.h>

#include "Convert.h"

#include <plier/tf/Convert.h>
#include <loco/IR/PermutingCodec.h>

#include <memory>
#include <cassert>

using namespace plier::tf;

namespace
{
using namespace moco;

class TFFakeQuantWithMinMaxVarsGraphUpdate final : public GraphUpdate
{
public:
  TFFakeQuantWithMinMaxVarsGraphUpdate(TFFakeQuantWithMinMaxVars *node,
                                       std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFFakeQuantWithMinMaxVars *_node;
  std::vector<TensorName> _names;
};

void TFFakeQuantWithMinMaxVarsGraphUpdate::input(const SymbolTable *node_table) const
{
  assert(_names.size() == 3);

  auto inputs_node = node_table->node(_names[0]);
  auto min_node = node_table->node(_names[1]);
  auto max_node = node_table->node(_names[2]);
  assert(inputs_node != nullptr);
  assert(min_node != nullptr);
  assert(max_node != nullptr);

  _node->inputs(inputs_node);
  _node->min(min_node);
  _node->max(max_node);
}

} // namespace

namespace moco
{

bool FakeQuantWithMinMaxVarsGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 3)
    return false;

  // attrs "narrow_range", "num_bits" are optional
  return true;
}

void FakeQuantWithMinMaxVarsGraphBuilder::build(const tensorflow::NodeDef &node,
                                                GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  auto fakequant_node = graph->nodes()->create<TFFakeQuantWithMinMaxVars>();
  fakequant_node->name(node.name());

  // read optional attributes
  if (has_attr(node, "num_bits"))
  {
    auto num_bits = get_int_attr(node, "num_bits");
    fakequant_node->num_bits(num_bits);
  }
  if (has_attr(node, "narrow_range"))
  {
    auto narrow_range = get_bool_attr(node, "narrow_range");
    fakequant_node->narrow_range(narrow_range);
  }

  // save the name for graph link updates
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, fakequant_node);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // inputs
  input_names.push_back(TensorName(node.input(1))); // min
  input_names.push_back(TensorName(node.input(2))); // max

  // Record ifm inputs to featureEncode_node
  auto tffakequant_update =
    std::make_unique<TFFakeQuantWithMinMaxVarsGraphUpdate>(fakequant_node, input_names);

  updates->enroll(std::move(tffakequant_update));
}

} // namespace moco
