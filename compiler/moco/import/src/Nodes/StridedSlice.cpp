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

#include "moco/Import/Nodes/StridedSlice.h"

#include <moco/IR/Nodes/TFStridedSlice.h>
#include <moco/IR/Nodes/TFConst.h>

#include <moco/Names.h>

#include "Convert.h"

#include <loco.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <memory>

namespace
{
using namespace moco;

class TFStridedSliceGraphUpdate final : public GraphUpdate
{
public:
  TFStridedSliceGraphUpdate(TFStridedSlice *node, std::vector<TensorName> names)
    : _node(node), _names(names)
  {
  }

  void input(const SymbolTable *) const override;

private:
  TFStridedSlice *_node;
  std::vector<TensorName> _names;
};

void TFStridedSliceGraphUpdate::input(const SymbolTable *node_table) const
{
  // TODO support size 3 where strides is None
  assert(_names.size() == 4);

  auto input_node = node_table->node(_names[0]);
  auto begin_node = node_table->node(_names[1]);
  auto end_node = node_table->node(_names[2]);
  auto strides_node = node_table->node(_names[3]);
  assert(input_node != nullptr);
  assert(begin_node != nullptr);
  assert(end_node != nullptr);
  assert(strides_node != nullptr);

  _node->input(input_node);
  _node->begin(begin_node);
  _node->end(end_node);
  _node->strides(strides_node);

  // TODO move validation codes to some suitable place
  // Run basic validation

  // TODO support full mask features
  if (_node->begin_mask() != 0 || _node->end_mask() != 0 || _node->ellipsis_mask() != 0 ||
      _node->new_axis_mask() != 0 || _node->shrink_axis_mask() != 1)
  {
    throw oops::UserExn("Mask attributes are not supported for now: ", _node->name());
  }

  // Only Const are supported for now
  auto const_input = dynamic_cast<moco::TFConst *>(_node->input());
  auto const_begin = dynamic_cast<moco::TFConst *>(_node->begin());
  auto const_end = dynamic_cast<moco::TFConst *>(_node->end());
  auto const_strides = dynamic_cast<moco::TFConst *>(_node->strides());
  if (const_input == nullptr || const_begin == nullptr || const_end == nullptr ||
      const_strides == nullptr)
  {
    throw oops::UserExn("Only Const inputs are supported for now: ", _node->name());
  }

  // TODO support S64
  if (const_begin->dtype() != loco::DataType::S32 || const_end->dtype() != loco::DataType::S32 ||
      const_strides->dtype() != loco::DataType::S32)
  {
    throw oops::UserExn("Only Const types of INT32 are supported for begin/end/strides for now: ",
                        _node->name());
  }

  // Input Rank should match number of elements of the begin/end/strides
  auto rin = const_input->rank();
  if (rin != const_begin->size<loco::DataType::S32>() ||
      rin != const_end->size<loco::DataType::S32>() ||
      rin != const_strides->size<loco::DataType::S32>())
  {
    throw oops::UserExn("Ranks for inputs should be same: ", _node->name());
  }

  // TODO support strides type of S64
  // TODO support other strides value
  // Only support stride 1 for now
  uint32_t elements = const_strides->size<loco::DataType::S32>();
  for (uint32_t e = 0; e < elements; ++e)
  {
    if (const_strides->at<loco::DataType::S32>(e) != 1)
    {
      throw oops::UserExn("Only stride 1 is supported for now: ", _node->name());
    }
  }
}

} // namespace

namespace moco
{

bool StridedSliceGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  // TODO support node.input_size() == 3 where strides is None
  if (node.input_size() != 4)
    return false;

  if (!plier::tf::has_attrs(node, {"T", "Index", "begin_mask", "end_mask", "ellipsis_mask",
                                   "new_axis_mask", "shrink_axis_mask"}))
    return false;

  return true;
}

void StridedSliceGraphBuilder::build(const tensorflow::NodeDef &node,
                                     GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  std::string node_name = node.name();

  auto stridedslice = graph->nodes()->create<TFStridedSlice>();
  stridedslice->name(node_name);

  // read attributes
  auto begin_mask = plier::tf::get_int_attr(node, "begin_mask");
  auto end_mask = plier::tf::get_int_attr(node, "end_mask");
  auto ellipsis_mask = plier::tf::get_int_attr(node, "ellipsis_mask");
  auto new_axis_mask = plier::tf::get_int_attr(node, "new_axis_mask");
  auto shrink_axis_mask = plier::tf::get_int_attr(node, "shrink_axis_mask");

  stridedslice->begin_mask(begin_mask);
  stridedslice->end_mask(end_mask);
  stridedslice->ellipsis_mask(ellipsis_mask);
  stridedslice->new_axis_mask(new_axis_mask);
  stridedslice->shrink_axis_mask(shrink_axis_mask);

  // TODO support general mask values: we support only this limited case for now
  assert(begin_mask == 0);
  assert(end_mask == 0);
  assert(ellipsis_mask == 0);
  assert(new_axis_mask == 0);
  assert(shrink_axis_mask == 1);

  // save the name for graph link updates
  TensorName output_name(node_name, 0);
  tensor_names->enroll(output_name, stridedslice);

  std::vector<TensorName> input_names;
  input_names.push_back(TensorName(node.input(0))); // input
  input_names.push_back(TensorName(node.input(1))); // begin
  input_names.push_back(TensorName(node.input(2))); // end
  input_names.push_back(TensorName(node.input(3))); // strides

  auto tfconv2d_update = std::make_unique<TFStridedSliceGraphUpdate>(stridedslice, input_names);

  updates->enroll(std::move(tfconv2d_update));
}

} // namespace moco
