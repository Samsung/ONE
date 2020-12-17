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

#include "moco/Import/Nodes/DepthwiseConv2dNative.h"

#include <moco/IR/Nodes/TFDepthwiseConv2dNative.h>

#include <moco/Names.h>

#include "Convert.h"

#include <plier/tf/Convert.h>
#include <loco/IR/PermutingCodec.h>
#include <stdex/Memory.h>
#include <oops/UserExn.h>

#include <cassert>

using namespace plier::tf;

namespace
{
using namespace moco;

class TFDepthwiseConv2dNativeGraphUpdate final : public GraphUpdate
{
public:
  TFDepthwiseConv2dNativeGraphUpdate(TFDepthwiseConv2dNative *node, std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFDepthwiseConv2dNative *_node;
  std::vector<TensorName> _names;
};

void TFDepthwiseConv2dNativeGraphUpdate::input(const SymbolTable *node_table) const
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

bool DepthwiseConv2dNativeGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 2)
    return false;

  // note: even though "data_format" and "dilations" are not entered when a model is written,
  //       TF seems to generate those field into a pb file.
  if (!has_attrs(node, {"T", "data_format", "dilations", "padding", "strides"}))
    return false;

  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  if (!(data_layout == "NHWC" || data_layout == "NCHW"))
  {
    throw oops::UserExn("DepthwiseConv2dNative Unsupported data_format", node.name());
  }

  auto padding = moco::str_toupper(get_string_attr(node, "padding"));
  if (!(padding == "VALID" || padding == "SAME"))
    return false;

  auto tf_strides = get_list_attr(node, "strides");
  auto strides = as_int64_list(tf_strides);
  if (!(strides.size() == 4))
  {
    throw oops::UserExn("DepthwiseConv2dNative strides requires rank 4", node.name());
  }
  auto stride_n = strides.at(0);
  auto stride_h = strides.at(1);
  auto stride_w = strides.at(2);
  auto stride_c = strides.at(3);
  if (!(stride_n == 1 && stride_c == 1) || !(stride_h == stride_w))
  {
    // TODO this message may need to be refined
    throw oops::UserExn("DepthwiseConv2dNative strides requires N=C=1, H=W", node.name());
  }

  return true;
}

void DepthwiseConv2dNativeGraphBuilder::build(const tensorflow::NodeDef &node,
                                              GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  auto depthwiseconv2d_native_node = graph->nodes()->create<TFDepthwiseConv2dNative>();
  depthwiseconv2d_native_node->name(node.name());

  // read attributes
  auto data_layout = get_string_attr(node, "data_format");
  depthwiseconv2d_native_node->data_layout(data_layout);

  auto tf_strides = get_list_attr(node, "strides");
  auto strides = as_int64_list(tf_strides);
  depthwiseconv2d_native_node->strides(strides);

  auto padding = moco::str_toupper(get_string_attr(node, "padding"));
  depthwiseconv2d_native_node->padding(padding);

  // save the name for graph link updates
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, depthwiseconv2d_native_node);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // input
  input_names.push_back(TensorName(node.input(1))); // kernel

  // Record ifm inputs to featureEncode_node
  auto tfdepthwiseconv2dnative_update = stdex::make_unique<TFDepthwiseConv2dNativeGraphUpdate>(
    depthwiseconv2d_native_node, input_names);

  updates->enroll(std::move(tfdepthwiseconv2dnative_update));
}

} // namespace moco
