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

#include "moco/Import/Nodes/BiasAdd.h"

#include <moco/IR/Nodes/TFBiasAdd.h>

#include <moco/Names.h>

#include <loco.h>
#include <loco/IR/PermutingCodec.h>
#include <stdex/Memory.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <cassert>
#include <vector>

namespace
{
using namespace moco;

class TFBiasAddGraphUpdate final : public GraphUpdate
{
public:
  TFBiasAddGraphUpdate(TFBiasAdd *biasadd, std::vector<TensorName> &names)
    : _biasadd(biasadd), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFBiasAdd *_biasadd;
  std::vector<TensorName> _names;
};

void TFBiasAddGraphUpdate::input(const SymbolTable *node_table) const
{
  assert(_names.size() == 2);

  auto value_node = node_table->node(_names[0]);
  auto bias_node = node_table->node(_names[1]);
  assert(value_node != nullptr);
  assert(bias_node != nullptr);

  _biasadd->value(value_node);
  _biasadd->bias(bias_node);
}

} // namespace

namespace moco
{

bool BiasAddGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (node.input_size() != 2)
    return false;

  // note: even though "data_format" is not entered when a model is written,
  //       TF seems to generate "data_format" field into a pb file
  if (!plier::tf::has_attrs(node, {"T", "data_format"}))
    return false;

  // TODO add type check
  // type of input and bias should be same (except using quantization)

  // Note In case of TF.nn.bias_add,
  // "value may have any number of dimensions." ...
  // but "data_format: A string. 'NHWC' and 'NCHW' are supported."
  // Not sure if value should be 4-D tensor. Let's skip this check for now.

  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  if (!(data_layout == "NHWC" || data_layout == "NCHW"))
  {
    throw oops::UserExn("BiasAdd Unsupported data_format", node.name());
  }

  return true;
}

void BiasAddGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // tensorflow data_format: one of NHWC or NCHW.
  auto data_layout = plier::tf::get_string_attr(node, "data_format");
  auto tf_bias_add = graph->nodes()->create<TFBiasAdd>();
  tf_bias_add->name(node.name());
  tf_bias_add->data_layout(data_layout);

  // To set the input node of encode_node with biasAdd_name
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_bias_add);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0)));
  input_names.push_back(TensorName(node.input(1)));

  auto update = stdex::make_unique<TFBiasAddGraphUpdate>(tf_bias_add, input_names);
  updates->enroll(std::move(update));
}

} // namespace moco
