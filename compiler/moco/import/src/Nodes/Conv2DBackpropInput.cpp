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

#include "moco/Import/Nodes/Conv2DBackpropInput.h"

#include <moco/IR/Nodes/TFConv2DBackpropInput.h>

#include "Convert.h"

#include <loco.h>
#include <stdex/Memory.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

namespace
{
using namespace moco;

/// @brief  GraphUpdate for Conv2DBackpropInput node
class Conv2DBackpropInputGraphUpdate final : public GraphUpdate
{
public:
  Conv2DBackpropInputGraphUpdate(TFConv2DBackpropInput *node, std::vector<TensorName> names)
    : _node(node), _input_names(names)
  {
    // DO NOTHING
  }

  void input(const SymbolTable *) const override;

private:
  TFConv2DBackpropInput *_node;
  std::vector<TensorName> _input_names;
};

void Conv2DBackpropInputGraphUpdate::input(const SymbolTable *table) const
{
  assert(_input_names.size() == 3);

  auto input_sizes_node = table->node(_input_names[0]);
  auto filter_node = table->node(_input_names[1]);
  auto out_backprop_node = table->node(_input_names[2]);

  assert(input_sizes_node != nullptr);
  assert(filter_node != nullptr);
  assert(out_backprop_node != nullptr);

  _node->input_sizes(input_sizes_node);
  _node->filter(filter_node);
  _node->out_backprop(out_backprop_node);
}

} // namespace

namespace moco
{

bool Conv2DBackpropInputGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 3)
    return false;

  if (!plier::tf::has_attrs(node, {"T", "data_format", "padding", "strides"}))
    return false;

  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  if (!(data_layout == "NHWC" || data_layout == "NCHW"))
  {
    throw oops::UserExn("Conv2DBackprop Unsupported data_format", node.name());
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

void Conv2DBackpropInputGraphBuilder::build(const tensorflow::NodeDef &node,
                                            GraphBuilderContext *context) const
{
  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // name of loco nodes
  std::string conv2d_backprop_name = node.name();

  auto conv2d_backprop = graph->nodes()->create<TFConv2DBackpropInput>();
  conv2d_backprop->name(node.name());

  // read attributes
  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  assert(data_layout == "NHWC" || data_layout == "NCHW");
  conv2d_backprop->data_layout(data_layout);

  auto tf_strides = plier::tf::get_list_attr(node, "strides");
  auto strides = plier::tf::as_int64_list(tf_strides);
  conv2d_backprop->strides(strides);

  auto padding = moco::str_toupper(plier::tf::get_string_attr(node, "padding"));
  assert(padding == "VALID" || padding == "SAME");
  conv2d_backprop->padding(padding);

  // save the name for graph link updates
  TensorName output_name(conv2d_backprop_name, 0);
  tensor_names->enroll(output_name, conv2d_backprop);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // input_sizes
  input_names.push_back(TensorName(node.input(1))); // filter
  input_names.push_back(TensorName(node.input(2))); // out_backprop

  // update
  auto conv2d_backprop_update =
    stdex::make_unique<Conv2DBackpropInputGraphUpdate>(conv2d_backprop, input_names);

  updates->enroll(std::move(conv2d_backprop_update));
}

} // namespace moco
