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

#include "moco/Import/Nodes/AvgPool.h"

#include <moco/IR/Nodes/TFAvgPool.h>

#include <moco/Names.h>

#include "Convert.h"
#include <loco/IR/PermutingCodec.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <memory>
#include <cassert>
#include <stdexcept>

using namespace plier::tf;

namespace
{

using namespace moco;

class TFAvgPoolGraphUpdate final : public GraphUpdate
{
public:
  TFAvgPoolGraphUpdate(TFAvgPool *node, const TensorName &name)
    : _avgpool_node(node), _value_name(name)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFAvgPool *_avgpool_node;
  const TensorName _value_name;
};

void TFAvgPoolGraphUpdate::input(const SymbolTable *node_table) const
{
  loco::Node *value_node = node_table->node(_value_name);
  _avgpool_node->value(value_node);
}

} // namespace

namespace moco
{

bool AvgPoolGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 1)
    return false;

  // note: even though "data_format" is not entered when a model is written,
  //       TF seems to generate "data_format" field into a pb file
  if (!plier::tf::has_attrs(node, {"T", "data_format", "ksize", "padding", "strides"}))
    return false;

  auto tf_ksize = get_list_attr(node, "ksize");
  auto ksize = as_int64_list(tf_ksize);
  if (ksize.size() != 4)
  {
    // TODO support ksize length for 1 and 2
    throw oops::UserExn("AvgPool only supports ksize length 4", node.name());
  }

  auto tf_strides = get_list_attr(node, "strides");
  auto strides = as_int64_list(tf_strides);
  if (strides.size() != 4)
  {
    // TODO support strides length for 1 and 2
    throw oops::UserExn("AvgPool only supports strides length 4", node.name());
  }

  return true;
}

void AvgPoolGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // name of loco nodes
  ::std::string avgPool2d_name = node.name();

  // tensorflow data_format: one of NHWC or NCHW.
  auto data_layout = get_string_attr(node, "data_format");
  auto avgPool_node = graph->nodes()->create<TFAvgPool>();
  avgPool_node->name(node.name());
  avgPool_node->data_layout(data_layout);

  // padding
  auto padding = moco::str_toupper(get_string_attr(node, "padding"));
  avgPool_node->padding(padding);

  // ksize
  auto tf_ksize = get_list_attr(node, "ksize");
  auto ksize = as_int64_list(tf_ksize);
  avgPool_node->ksize(ksize);

  // strides
  auto tf_strides = get_list_attr(node, "strides");
  auto strides = as_int64_list(tf_strides);
  avgPool_node->strides(strides);

  // To set the input node of encode_node with avgPool2d_name
  TensorName output_name(avgPool2d_name, 0);
  tensor_names->enroll(output_name, avgPool_node);

  // Record ifm inputs to featureEncode_node
  auto update = std::make_unique<TFAvgPoolGraphUpdate>(avgPool_node, TensorName(node.input(0)));

  updates->enroll(std::move(update));
}

} // namespace moco

// TODO Consider a case when TF AvgPool is for 3D.
// AvgPool works for 2D and other Dimensions, such as 3D
// So, in future, some other GraphBuilder decide if AvgPoolGraphBuilder is used or
// other GraphBuilder is used for TF AvgPool
