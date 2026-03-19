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

#include "moco/Import/Nodes/Conv2D.h"

#include <moco/IR/Nodes/TFConv2D.h>

#include <moco/Names.h>

#include "Convert.h"

#include <loco.h>
#include <loco/IR/PermutingCodec.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <memory>
#include <cassert>
#include <stdexcept>
#include <algorithm>

namespace
{
using namespace moco;

class TFConv2DGraphUpdate final : public GraphUpdate
{
public:
  TFConv2DGraphUpdate(TFConv2D *node, std::vector<TensorName> names) : _node(node), _names(names) {}

  void input(const SymbolTable *) const override;

private:
  TFConv2D *_node;
  std::vector<TensorName> _names;
};

void TFConv2DGraphUpdate::input(const SymbolTable *node_table) const
{
  assert(_names.size() == 2);

  auto input_node = node_table->node(_names[0]);
  auto filter_node = node_table->node(_names[1]);
  assert(input_node != nullptr);
  assert(filter_node != nullptr);

  _node->input(input_node);
  _node->filter(filter_node);
}

} // namespace

namespace moco
{

bool Conv2DGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 2)
    return false;

  // note: even though "data_format" is not entered when a model is written,
  //       TF seems to generate "data_format" field into a pb file
  if (!plier::tf::has_attrs(node, {"T", "data_format", "padding", "strides"}))
    return false;

  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  if (!(data_layout == "NHWC" || data_layout == "NCHW"))
  {
    throw oops::UserExn("Conv2D Unsupported data_format", node.name());
  }

  // dilation attribute is not fully supported
  if (plier::tf::has_attr(node, "dilations"))
  {
    // TODO Support non-default dilations
    auto dilation = plier::tf::get_list_attr(node, "dilations").i();
    if (!std::all_of(dilation.begin(), dilation.end(), [](std::int64_t dil) { return dil == 1; }))
      return false;
  }
  // Else, dilations are automatically set to default [1,1,1,1] which we assumes now

  return true;
}

void Conv2DGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // name of loco nodes
  std::string conv2d_name = node.name();

  auto conv2d = graph->nodes()->create<TFConv2D>();
  conv2d->name(node.name());

  // read attributes
  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  assert(data_layout == "NHWC" || data_layout == "NCHW");
  conv2d->data_layout(data_layout);

  auto tf_strides = plier::tf::get_list_attr(node, "strides");
  auto strides = plier::tf::as_int64_list(tf_strides);
  conv2d->strides(strides);

  auto padding = moco::str_toupper(plier::tf::get_string_attr(node, "padding"));
  assert(padding == "VALID" || padding == "SAME");
  conv2d->padding(padding);

  // save the name for graph link updates
  TensorName output_name(conv2d_name, 0);
  tensor_names->enroll(output_name, conv2d);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // input
  input_names.push_back(TensorName(node.input(1))); // kernel

  // Record ifm inputs to featureEncode_node
  auto tfconv2d_update = std::make_unique<TFConv2DGraphUpdate>(conv2d, input_names);

  updates->enroll(std::move(tfconv2d_update));
}

} // namespace moco
