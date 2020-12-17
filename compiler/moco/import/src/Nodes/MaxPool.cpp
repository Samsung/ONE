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

#include "moco/Import/Nodes/MaxPool.h"

#include <moco/IR/Nodes/TFMaxPool.h>

#include <moco/Names.h>

#include "Convert.h"

#include <loco.h>
#include <loco/IR/PermutingCodec.h>
#include <stdex/Memory.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <cassert>
#include <stdexcept>

namespace
{

using namespace moco;

class TFMaxPoolGraphUpdate final : public GraphUpdate
{
public:
  TFMaxPoolGraphUpdate(TFMaxPool *node, const TensorName &name)
    : _maxpool_node(node), _input_name(name)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFMaxPool *_maxpool_node;
  const TensorName _input_name;
};

void TFMaxPoolGraphUpdate::input(const SymbolTable *node_table) const
{
  loco::Node *input_node = node_table->node(_input_name);
  _maxpool_node->input(input_node);
}

} // namespace

namespace moco
{

bool MaxPoolGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  // note: even though "data_format" is not entered when a model is written,
  //       TF seems to generate "data_format" field into a pb file
  if (!plier::tf::has_attrs(node, {"T", "data_format", "ksize", "padding", "strides"}))
    return false;

  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  if (!(data_layout == "NHWC" || data_layout == "NCHW"))
  {
    throw oops::UserExn("MaxPool Unsupported data_format", node.name());
  }

  auto tf_ksize = plier::tf::get_list_attr(node, "ksize");
  auto ksize = plier::tf::as_int64_list(tf_ksize);
  if (ksize.size() != 4)
  {
    // TODO support ksize length for 1 and 2
    throw oops::UserExn("MaxPool ksize requires rank 4", node.name());
  }

  auto tf_strides = plier::tf::get_list_attr(node, "strides");
  auto strides = plier::tf::as_int64_list(tf_strides);
  if (strides.size() != 4)
  {
    // TODO support strides length for 1 and 2
    throw oops::UserExn("MaxPool strides requires rank 4", node.name());
  }

  return true;
}

void MaxPoolGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // name of loco nodes
  ::std::string node_name = node.name();

  // tensorflow data_format: one of NHWC or NCHW.
  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  auto maxPool_node = graph->nodes()->create<TFMaxPool>();
  maxPool_node->name(node.name());
  maxPool_node->data_layout(data_layout);

  // padding
  auto padding = moco::str_toupper(plier::tf::get_string_attr(node, "padding"));
  maxPool_node->padding(padding);

  // ksize
  auto tf_ksize = plier::tf::get_list_attr(node, "ksize");
  auto ksize = plier::tf::as_int64_list(tf_ksize);
  assert(ksize.size() == 4);
  maxPool_node->ksize(ksize);

  // strides
  auto tf_strides = plier::tf::get_list_attr(node, "strides");
  auto strides = plier::tf::as_int64_list(tf_strides);
  assert(strides.size() == 4);
  maxPool_node->strides(strides);

  // To set the input node of encode_node with node_name
  TensorName output_name(node_name, 0);
  tensor_names->enroll(output_name, maxPool_node);

  // Record ifm inputs to featureEncode_node
  auto update = stdex::make_unique<TFMaxPoolGraphUpdate>(maxPool_node, TensorName(node.input(0)));

  updates->enroll(std::move(update));
}

} // namespace moco

// TODO Consider a case when TF MaxPool is for 3D.
// MaxPool works for 2D and other Dimensions, such as 3D
// So, in future, some other GraphBuilder decide if MaxPoolGraphBuilder is used or
// other GraphBuilder is used for TF MaxPool
