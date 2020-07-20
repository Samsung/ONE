/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleExporterUtils.h"

#include <oops/InternalExn.h>

#include <cassert>
#include <memory>

namespace luci
{

circle::ActivationFunctionType to_circle_actfunc(luci::FusedActFunc func)
{
  switch (func)
  {
    case luci::FusedActFunc::NONE:
      return circle::ActivationFunctionType_NONE;
    case luci::FusedActFunc::RELU:
      return circle::ActivationFunctionType_RELU;
    case luci::FusedActFunc::RELU_N1_TO_1:
      return circle::ActivationFunctionType_RELU_N1_TO_1;
    case luci::FusedActFunc::RELU6:
      return circle::ActivationFunctionType_RELU6;
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::FusedActFunc", oops::to_uint32(func));
  }
}

circle::TensorType to_circle_tensortype(loco::DataType type)
{
  switch (type)
  {
    case loco::DataType::U8:
      return circle::TensorType_UINT8;

    case loco::DataType::S8:
      return circle::TensorType_INT8;
    case loco::DataType::S16:
      return circle::TensorType_INT16;
    case loco::DataType::S32:
      return circle::TensorType_INT32;
    case loco::DataType::S64:
      return circle::TensorType_INT64;

    case loco::DataType::FLOAT16:
      return circle::TensorType_FLOAT16;
    case loco::DataType::FLOAT32:
      return circle::TensorType_FLOAT32;

    case loco::DataType::BOOL:
      return circle::TensorType_BOOL;

    default:
      INTERNAL_EXN_V("failed to convert unsupported loco::DataType", oops::to_uint32(type));
  }
}

circle::MirrorPadMode to_circle_mirrorpadmode(luci::MirrorPadMode mode)
{
  switch (mode)
  {
    case luci::MirrorPadMode::REFLECT:
      return circle::MirrorPadMode::MirrorPadMode_REFLECT;
    case luci::MirrorPadMode::SYMMETRIC:
      return circle::MirrorPadMode::MirrorPadMode_SYMMETRIC;
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::MirrorPadMode", oops::to_uint32(mode));
  }
}

} // namespace luci

namespace luci
{

uint32_t SerializedModelData::registerBuiltinOpcode(circle::BuiltinOperator builtin_code,
                                                    const int32_t op_version = 0)
{
  assert(op_version > 0);

  auto it = _operator_codes.find(OpCode{builtin_code, "", op_version});
  if (it != _operator_codes.end())
  {
    return it->second;
  }
  auto idx = static_cast<uint32_t>(_operator_codes.size());
  _operator_codes.emplace(OpCode{builtin_code, "", op_version}, idx);
  return idx;
}

uint32_t SerializedModelData::registerCustomOpcode(const std::string &custom_code)
{
  const circle::BuiltinOperator builtin_code = circle::BuiltinOperator_CUSTOM;
  auto it = _operator_codes.find(OpCode{builtin_code, custom_code});
  if (it != _operator_codes.end())
  {
    return it->second;
  }
  auto idx = static_cast<uint32_t>(_operator_codes.size());
  _operator_codes.emplace(OpCode{builtin_code, custom_code}, idx);
  return idx;
}

circle::Padding getOpPadding(const loco::Padding2D *pad, const loco::Stride<2> *stride,
                             const ShapeDescription &ifm, const ShapeDescription &ofm)
{
  // VALID padding
  if (pad->top() == 0 && pad->bottom() == 0 && pad->left() == 0 && pad->right() == 0)
    return circle::Padding_VALID;

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
    return circle::Padding_SAME;

  INTERNAL_EXN("Unsupported padding criteria");
}

circle::Padding getOpPadding(const luci::Padding pad)
{
  if (pad == luci::Padding::VALID)
    return circle::Padding_VALID;
  if (pad == luci::Padding::SAME)
    return circle::Padding_SAME;

  INTERNAL_EXN_V("Unsupported luci::Padding", oops::to_uint32(pad));
}

namespace
{

class CircleTensorIndexAnnotation final : public loco::NodeAnnotation
{
public:
  CircleTensorIndexAnnotation(const CircleTensorIndex &index) : _index{index}
  {
    // DO NOTHING
  }

public:
  const CircleTensorIndex &index(void) const { return _index; }

private:
  CircleTensorIndex _index;
};

} // namespace

void set_tensor_index(loco::Node *node, const CircleTensorIndex &tensor_id)
{
  assert(node->annot<CircleTensorIndexAnnotation>() == nullptr);
  node->annot(std::make_unique<CircleTensorIndexAnnotation>(tensor_id));
}

CircleTensorIndex get_tensor_index(loco::Node *node)
{
  assert(node->annot<CircleTensorIndexAnnotation>() != nullptr);
  return node->annot<CircleTensorIndexAnnotation>()->index();
}

} // namespace luci
