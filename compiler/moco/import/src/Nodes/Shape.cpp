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

#include "moco/Import/Nodes/Shape.h"

#include <moco/IR/Nodes/TFShape.h>

#include <loco.h>
#include <plier/tf/Convert.h>

#include <memory>

namespace
{
using namespace moco;

/**
 * @brief GraphUpdate for Shape node
 */
class ShapeGraphUpdate final : public GraphUpdate
{
public:
  ShapeGraphUpdate(TFShape *node, const TensorName &&input_name)
    : _node(node), _input_name(input_name)
  {
    // DO NOTHING
  }

  void input(const SymbolTable *) const override;

private:
  TFShape *_node;
  const TensorName _input_name;
};

void ShapeGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *input_node = table->node(_input_name);
  _node->input(input_node);
}

} // namespace

namespace moco
{

bool ShapeGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 1)
    return false;

  return plier::tf::has_attrs(node, {"T"});
}

void ShapeGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // create TF dialect Shape node
  auto tf_shape = graph->nodes()->create<TFShape>();
  tf_shape->name(node.name());

  if (plier::tf::has_attrs(node, {"out_type"}))
  {
    auto dtype = plier::tf::as_loco_datatype(plier::tf::get_datatype_attr(node, "out_type"));
    // TODO Support other dtype like S64
    assert(dtype == loco::DataType::S32);

    tf_shape->dtype(dtype);
  }
  else
  {
    // Set to S32, TF-documented default value for 'out_type'
    tf_shape->dtype(loco::DataType::S32);
  }

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_shape);

  auto update = std::make_unique<ShapeGraphUpdate>(tf_shape, TensorName(node.input(0)));
  updates->enroll(std::move(update));
}

} // namespace moco
