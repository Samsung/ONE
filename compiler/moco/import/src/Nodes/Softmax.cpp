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

#include "moco/Import/Nodes/Softmax.h"

#include <moco/IR/Nodes/TFSoftmax.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>

namespace
{
using namespace moco;

/**
 * @brief GraphUpdate for Softmax node
 */
class SoftmaxGraphUpdate final : public GraphUpdate
{
public:
  SoftmaxGraphUpdate(TFSoftmax *node, const TensorName &&input_name)
    : _node(node), _input_name(input_name)
  {
    // DO NOTHING
  }

  void input(const SymbolTable *) const override;

private:
  TFSoftmax *_node;
  const TensorName _input_name;
};

void SoftmaxGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *input_node = table->node(_input_name);
  _node->logits(input_node);
}

} // namespace

namespace moco
{

bool SoftmaxGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 1)
    return false;

  return plier::tf::has_attrs(node, {"T"});
}

void SoftmaxGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect Softmax node
  auto tf_softmax = graph->nodes()->create<TFSoftmax>();
  tf_softmax->name(node.name());

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_softmax);

  auto update = std::make_unique<SoftmaxGraphUpdate>(tf_softmax, TensorName(node.input(0)));
  updates->enroll(std::move(update));
}

} // namespace moco
