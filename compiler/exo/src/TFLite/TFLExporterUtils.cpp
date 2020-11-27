/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TFLExporterUtils.h"

#include <oops/InternalExn.h>

namespace exo
{

tflite::ActivationFunctionType to_tflite_actfunc(locoex::FusedActFunc func)
{
  switch (func)
  {
    case locoex::FusedActFunc::NONE:
      return tflite::ActivationFunctionType_NONE;
    case locoex::FusedActFunc::RELU:
      return tflite::ActivationFunctionType_RELU;
    case locoex::FusedActFunc::RELU6:
      return tflite::ActivationFunctionType_RELU6;
    default:
      INTERNAL_EXN_V("Unsupported locoex FusedActFunc Type", oops::to_uint32(func));
  }
}

} // namespace exo

namespace exo
{
namespace tflite_detail
{

uint32_t SerializedModelData::registerBuiltinOpcode(tflite::BuiltinOperator builtin_code)
{
  auto it = _operator_codes.find(OpCode{builtin_code});
  if (it != _operator_codes.end())
  {
    return it->second;
  }
  auto idx = static_cast<uint32_t>(_operator_codes.size());
  _operator_codes.emplace(OpCode{builtin_code}, idx);
  return idx;
}

uint32_t SerializedModelData::registerCustomOpcode(const std::string &custom_op)
{
  tflite::BuiltinOperator custom_code = tflite::BuiltinOperator_CUSTOM;
  auto idx = registerBuiltinOpcode(custom_code);
  _custom_operator_codes.emplace(OpCode{custom_code}, custom_op);
  return idx;
}

tflite::Padding getOpPadding(const loco::Padding2D *pad, const loco::Stride<2> *stride,
                             const ShapeDescription &ifm, const ShapeDescription &ofm)
{
  // VALID padding
  if (pad->top() == 0 && pad->bottom() == 0 && pad->left() == 0 && pad->right() == 0)
    return tflite::Padding_VALID;

  // SAME padding
  //
  // For same padding, by definition, following equation should hold:
  //   O = floor((I - 1) / S) + 1
  //   where input size I, output size O, stride S
  //
  // NOTE input and output 'feature' map are shape of NHWC
  bool same_padding_criterion_1 =
    (static_cast<uint32_t>(ofm._dims[1]) == (ifm._dims[1] - 1) / stride->vertical() + 1) &&
    (static_cast<uint32_t>(ofm._dims[2]) == (ifm._dims[2] - 1) / stride->horizontal() + 1);

  // For same padding, rear padding is same or bigger than front padding by at most 1
  bool same_padding_criterion_2 =
    (pad->top() <= pad->bottom()) && (pad->bottom() <= pad->top() + 1) &&
    (pad->left() <= pad->right()) && (pad->right() <= pad->left() + 1);

  if (same_padding_criterion_1 && same_padding_criterion_2)
    return tflite::Padding_SAME;

  INTERNAL_EXN("NYI for custom PAD");
}

tflite::Padding getOpPadding(const locoex::Padding pad)
{
  if (pad == locoex::Padding::VALID)
    return tflite::Padding_VALID;
  if (pad == locoex::Padding::SAME)
    return tflite::Padding_SAME;

  INTERNAL_EXN_V("Unknown padding", oops::to_uint32(pad));
}

void registerGraphIOName(loco::Graph *graph, SerializedModelData &gd)
{
  for (uint32_t in = 0; in < graph->inputs()->size(); ++in)
  {
    auto pull = loco::pull_node(graph, in);
    auto name = graph->inputs()->at(in)->name();

    gd._pull_to_name[pull] = name;
  }
  for (uint32_t out = 0; out < graph->outputs()->size(); ++out)
  {
    auto push = loco::push_node(graph, out);
    auto name = graph->outputs()->at(out)->name();

    gd._push_to_name[push] = name;
  }
}

#include <stdex/Memory.h>

#include <cassert>

namespace
{

class TFLTensorIndexAnnotation final : public loco::NodeAnnotation
{
public:
  TFLTensorIndexAnnotation(const TFLTensorIndex &index) : _index{index}
  {
    // DO NOTHING
  }

public:
  const TFLTensorIndex &index(void) const { return _index; }

private:
  TFLTensorIndex _index;
};

} // namespace

void set_tensor_index(loco::Node *node, const TFLTensorIndex &tensor_id)
{
  assert(node->annot<TFLTensorIndexAnnotation>() == nullptr);
  node->annot(stdex::make_unique<TFLTensorIndexAnnotation>(tensor_id));
}

TFLTensorIndex get_tensor_index(loco::Node *node)
{
  assert(node->annot<TFLTensorIndexAnnotation>() != nullptr);
  return node->annot<TFLTensorIndexAnnotation>()->index();
}

} // namespace tflite_detail
} // namespace exo
