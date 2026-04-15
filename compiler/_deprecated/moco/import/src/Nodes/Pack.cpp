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

#include "moco/Import/Nodes/Pack.h"

#include <moco/IR/Nodes/TFPack.h>
#include <moco/IR/Nodes/TFConst.h>

#include <moco/Names.h>

#include <loco.h>
#include <loco/IR/NodeShape.h>
#include <plier/tf/Convert.h>

#include <memory>
#include <cassert>

namespace
{

using namespace moco;

class TFPackGraphUpdate final : public GraphUpdate
{
public:
  TFPackGraphUpdate(TFPack *node, std::vector<TensorName> names) : _node(node), _names(names) {}

  void input(const SymbolTable *) const override;

private:
  TFPack *_node;
  std::vector<TensorName> _names;
};

void TFPackGraphUpdate::input(const SymbolTable *tensor_names) const
{
  uint32_t num_values = _names.size();
  assert(num_values >= 1);

  for (uint32_t i = 0; i < num_values; ++i)
  {
    auto input_node = tensor_names->node(_names[i]);
    assert(input_node != nullptr);
    _node->values(i, input_node);
  }
}

} // namespace

namespace moco
{

bool PackGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (!plier::tf::has_attrs(node, {"T", "N", "axis"}))
    return false;

  const int num_inputs = node.input_size();
  return (num_inputs >= 1) && (num_inputs == plier::tf::get_int_attr(node, "N"));
}

void PackGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();
  auto tensor_names = context->tensor_names();
  auto updates = context->updates();

  const int num_inputs = node.input_size();
  std::vector<TensorName> input_names;
  auto pack_node = graph->nodes()->create<TFPack>(num_inputs);
  pack_node->name(node.name());

  for (int ni = 0; ni < num_inputs; ++ni)
  {
    input_names.push_back(TensorName(node.input(ni)));
  }

  pack_node->axis(plier::tf::get_int_attr(node, "axis"));

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, pack_node);

  auto update = std::make_unique<TFPackGraphUpdate>(pack_node, input_names);
  updates->enroll(std::move(update));
}

} // namespace moco
