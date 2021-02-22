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

#include "COpCall.h"

#include "Convert.h"

#include <locoex/COpCall.h>
#include <locoex/COpAttrTypes.h>
#include <moco/Names.h>
#include <moco/tf/Frontend.h>
#include <loco.h>
#include <oops/UserExn.h>

#include <memory>
#include <vector>
#include <cassert>
#include <stdexcept>

namespace
{

class COpCallGraphUpdate final : public moco::GraphUpdate
{
public:
  COpCallGraphUpdate(locoex::COpCall *node, const std::vector<moco::TensorName> &input_names)
    : _node(node), _input_names(input_names)
  {
  }

  void input(const moco::SymbolTable *) const override;

private:
  locoex::COpCall *_node;
  const std::vector<moco::TensorName> _input_names;
};

void COpCallGraphUpdate::input(const moco::SymbolTable *tensor_names) const
{
  for (int n = 0; n < _input_names.size(); n++)
  {
    loco::Node *target = tensor_names->node(_input_names.at(n));
    _node->input(n, target);
  }
}

} // namespace

namespace moco
{
namespace tf
{

bool COpCallGraphBuilder::validate(const tensorflow::NodeDef &tf_node) const { return true; }

void COpCallGraphBuilder::build(const tensorflow::NodeDef &tf_node,
                                GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // Create a "COpCall" node for CustomOp and set attributes
  auto call_node = graph->nodes()->create<locoex::COpCall>(tf_node.input_size());
  {
    call_node->op(tf_node.op());
    call_node->name(tf_node.name());
    call_node->dtype(_signature->dtype(tf_node.name()));

    auto shape = _signature->shape(tf_node.name());
    call_node->rank(shape->rank());
    for (int d = 0; d < shape->rank(); d++)
      call_node->dim(d) = shape->dim(d);

    for (auto iter = tf_node.attr().begin(); iter != tf_node.attr().end(); iter++)
    {
      auto name = iter->first;
      auto val = iter->second;

      if (val.value_case() == tensorflow::AttrValue::kF)
      {
        call_node->attr(name, std::make_unique<locoex::COpAttrFloat>(val.f()));
      }
      else if (val.value_case() == tensorflow::AttrValue::kI)
      {
        call_node->attr(name, std::make_unique<locoex::COpAttrInt>(val.i()));
      }
      // TODO define more types
      else
      {
        throw oops::UserExn("Unsupported attribute type", tf_node.name());
      }
    }
  }

  // register this node with its name
  TensorName output_name(tf_node.name(), 0);
  tensor_names->enroll(output_name, call_node);

  // Queue node input update
  std::vector<TensorName> input_names;
  for (int i = 0; i < tf_node.input_size(); ++i)
  {
    input_names.emplace_back(TensorName(tf_node.input(i)));
  }
  auto update = std::make_unique<COpCallGraphUpdate>(call_node, input_names);
  updates->enroll(std::move(update));
}

} // namespace tf
} // namespace moco
