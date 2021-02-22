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

#include "moco/Import/Nodes/StopGradient.h"

#include <moco/IR/Nodes/TFStopGradient.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for TF StopGradient node
 */
class TFStopGradientGraphUpdate final : public GraphUpdate
{
public:
  TFStopGradientGraphUpdate(TFStopGradient *node, TensorName &&name) : _node(node), _name(name) {}

  void input(const SymbolTable *) const override;

private:
  TFStopGradient *_node;
  TensorName _name;
};

void TFStopGradientGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *target = table->node(_name);
  _node->input(target);
}

} // namespace

namespace moco
{

bool StopGradientGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 1)
    return false;

  return plier::tf::has_attrs(node, {"T"});
}

void StopGradientGraphBuilder::build(const tensorflow::NodeDef &node,
                                     GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect StopGradient node
  auto tf_stopgradient = graph->nodes()->create<TFStopGradient>();
  tf_stopgradient->name(node.name());

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_stopgradient);

  // Queue node input update
  auto tf_stopgradient_update =
    std::make_unique<TFStopGradientGraphUpdate>(tf_stopgradient, TensorName(node.input(0)));
  updates->enroll(std::move(tf_stopgradient_update));
}

} // namespace moco
