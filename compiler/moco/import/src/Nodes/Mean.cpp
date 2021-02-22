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

#include "moco/Import/Nodes/Mean.h"

#include <moco/IR/Nodes/TFMean.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>

namespace
{
using namespace moco;

/**
 * @brief GraphUpdate for Mean node
 */
class MeanGraphUpdate final : public GraphUpdate
{
public:
  MeanGraphUpdate(TFMean *node, const TensorName &&input_name,
                  const TensorName &&reduction_indices_name)
    : _node(node), _input_name(input_name), _reduction_indices_name(reduction_indices_name)
  {
    // DO NOTHING
  }

  void input(const SymbolTable *) const override;

private:
  TFMean *_node;
  const TensorName _input_name;
  const TensorName _reduction_indices_name;
};

void MeanGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *input_node = table->node(_input_name);
  loco::Node *reduction_indices_node = table->node(_reduction_indices_name);
  _node->input(input_node);
  _node->reduction_indices(reduction_indices_node);
}

} // namespace

namespace moco
{

bool MeanGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 2)
    return false;

  if (!plier::tf::has_attrs(node, {"T", "Tidx", "keep_dims"}))
    return false;

  auto dtype = plier::tf::get_datatype_attr(node, "Tidx");
  if (dtype != tensorflow::DataType::DT_INT32 && dtype != tensorflow::DataType::DT_INT64)
    return false;

  return true;
}

void MeanGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect Mean node
  auto tf_mean = graph->nodes()->create<TFMean>();
  tf_mean->name(node.name());
  tf_mean->keep_dims(plier::tf::get_bool_attr(node, "keep_dims"));

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_mean);

  auto update = std::make_unique<MeanGraphUpdate>(tf_mean, TensorName(node.input(0)),
                                                  TensorName(node.input(1)));
  updates->enroll(std::move(update));
}

} // namespace moco
