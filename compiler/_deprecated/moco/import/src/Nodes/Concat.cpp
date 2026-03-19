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

#include "moco/Import/Nodes/Concat.h"

#include <moco/IR/Nodes/TFConcatV2.h>

#include <moco/Names.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>
#include <cassert>

namespace
{

using namespace moco;

class TFConcatV2GraphUpdate final : public GraphUpdate
{
public:
  TFConcatV2GraphUpdate(TFConcatV2 *node, std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFConcatV2 *_node;
  std::vector<TensorName> _names;
};

void TFConcatV2GraphUpdate::input(const SymbolTable *tensor_names) const
{
  uint32_t num_values = _names.size() - 1; // exclude axis
  assert(num_values >= 1);

  for (uint32_t i = 0; i < num_values; ++i)
  {
    auto input_node = tensor_names->node(_names[i]);
    assert(input_node != nullptr);
    _node->values(i, input_node);
  }
  auto axis_node = tensor_names->node(_names[num_values]);
  assert(axis_node != nullptr);
  _node->axis(axis_node);
}

} // namespace

namespace moco
{

bool ConcatV2GraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (!plier::tf::has_attrs(node, {"T", "N", "Tidx"}))
    return false;

  // Concat node SHOULD have 3 or more inputs, that is 2 + axis
  const int num_inputs = node.input_size() - 1;
  return (num_inputs >= 2) && (num_inputs == plier::tf::get_int_attr(node, "N"));
}

void ConcatV2GraphBuilder::build(const tensorflow::NodeDef &node,
                                 GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();
  auto tensor_names = context->tensor_names();
  auto updates = context->updates();

  const int num_inputs = node.input_size() - 1;
  std::vector<TensorName> input_names;
  auto concat_node = graph->nodes()->create<TFConcatV2>(num_inputs);
  concat_node->name(node.name());

  for (int ni = 0; ni < num_inputs; ++ni)
  {
    input_names.push_back(TensorName(node.input(ni)));
  }
  // last one is the axis
  input_names.push_back(TensorName(node.input(num_inputs)));

  // register string-name to the last node as output of concat(s)
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, concat_node);

  auto update = std::make_unique<TFConcatV2GraphUpdate>(concat_node, input_names);
  updates->enroll(std::move(update));
}

} // namespace moco
