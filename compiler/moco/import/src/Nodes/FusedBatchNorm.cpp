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

#include "moco/Import/Nodes/FusedBatchNorm.h"

#include <moco/IR/Nodes/TFFusedBatchNorm.h>

#include <loco.h>
#include <stdex/Memory.h>
#include <plier/tf/Convert.h>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for FusedBatchNorm node
 */
class FusedBatchNormGraphUpdate final : public GraphUpdate
{
public:
  FusedBatchNormGraphUpdate(TFFusedBatchNorm *node, std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFFusedBatchNorm *_node;
  std::vector<TensorName> _names;
};

void FusedBatchNormGraphUpdate::input(const SymbolTable *tensor_names) const
{
  assert(_names.size() == 5);

  _node->x(tensor_names->node(_names[0]));
  _node->scale(tensor_names->node(_names[1]));
  _node->offset(tensor_names->node(_names[2]));
  _node->mean(tensor_names->node(_names[3]));
  _node->variance(tensor_names->node(_names[4]));
}

} // namespace

namespace moco
{

bool FusedBatchNormGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 5)
    return false;

  return plier::tf::has_attrs(node, {"epsilon"});
}

void FusedBatchNormGraphBuilder::build(const tensorflow::NodeDef &node,
                                       GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  float epsilon = plier::tf::get_float_attr(node, "epsilon");

  // creating TF dialect FusedBatchNorm node
  auto tf_fbn = graph->nodes()->create<TFFusedBatchNorm>();
  tf_fbn->name(node.name());
  tf_fbn->epsilon(epsilon);

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_fbn);

  std::vector<TensorName> fbn_input_names;
  fbn_input_names.push_back(TensorName(node.input(0))); // input
  fbn_input_names.push_back(TensorName(node.input(1))); // scale
  fbn_input_names.push_back(TensorName(node.input(2))); // offset
  fbn_input_names.push_back(TensorName(node.input(3))); // mean
  fbn_input_names.push_back(TensorName(node.input(4))); // variance

  auto tf_fbn_update = stdex::make_unique<FusedBatchNormGraphUpdate>(tf_fbn, fbn_input_names);
  updates->enroll(std::move(tf_fbn_update));
}

} // namespace moco
