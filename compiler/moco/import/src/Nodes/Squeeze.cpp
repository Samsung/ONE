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

#include "moco/Import/Nodes/Squeeze.h"

#include <moco/IR/Nodes/TFSqueeze.h>

#include <moco/Names.h>

#include <loco.h>
#include <stdex/Memory.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

namespace
{
using namespace moco;

/**
 * @brief GraphUpdate for Squeeze node
 */
class SqueezeGraphUpdate final : public GraphUpdate
{
public:
  SqueezeGraphUpdate(TFSqueeze *node, const TensorName &&input_name)
    : _node(node), _input_name(input_name)
  {
    // DO NOTHING
  }

  void input(const SymbolTable *) const override;

private:
  TFSqueeze *_node;
  const TensorName _input_name;
};

void SqueezeGraphUpdate::input(const SymbolTable *table) const
{
  loco::Node *input_node = table->node(_input_name);
  _node->input(input_node);
}

} // namespace

namespace moco
{

bool SqueezeGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 1)
    return false;

  if (!plier::tf::has_attrs(node, {"T"}))
    return false;

  if (plier::tf::has_attrs(node, {"axis"}))
  {
    // TODO support 'axis' attribute
    oops::UserExn("Squeeze: Unsupported 'axis' attribute", node.name());
  }

  return true;
}

void SqueezeGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // TODO support 'axis' attribute
  assert(!plier::tf::has_attrs(node, {"axis"}));

  std::vector<int64_t> squeeze_dims;
  if (plier::tf::has_attrs(node, {"squeeze_dims"}))
  {
    auto squeeze_dim_list = plier::tf::get_list_attr(node, {"squeeze_dims"});
    // TODO assert squeeze_dims are mutually different?
    squeeze_dims = plier::tf::as_int64_list(squeeze_dim_list);
  }
  // Note that it is possible that NodeDef does not have squeeze_dims attribute.
  // In that case, TFSqueeze also has empty squeeze_dims,

  // creating TF dialect Squeeze node
  auto tf_squeeze = graph->nodes()->create<TFSqueeze>();
  tf_squeeze->name(node.name());
  tf_squeeze->squeeze_dims(squeeze_dims);

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_squeeze);

  auto update = stdex::make_unique<SqueezeGraphUpdate>(tf_squeeze, TensorName(node.input(0)));
  updates->enroll(std::move(update));
}

} // namespace moco
