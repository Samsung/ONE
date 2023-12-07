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
#include "CircleBuiltinTypesMappingRule.h"

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
    case luci::FusedActFunc::TANH:
      return circle::ActivationFunctionType_TANH;
    case luci::FusedActFunc::SIGN_BIT:
      return circle::ActivationFunctionType_SIGN_BIT;
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

    case loco::DataType::S4:
      return circle::TensorType_INT4;
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

    case loco::DataType::STRING:
      return circle::TensorType_STRING;

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

circle::FullyConnectedOptionsWeightsFormat
to_circle_weightsformat(luci::CircleFullyConnected::WeightsFormat format)
{
  switch (format)
  {
    case luci::CircleFullyConnected::WeightsFormat::DEFAULT:
      return circle::FullyConnectedOptionsWeightsFormat_DEFAULT;
    case luci::CircleFullyConnected::WeightsFormat::SHUFFLED4x16INT8:
      return circle::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;
    case luci::CircleFullyConnected::WeightsFormat::SHUFFLED16x1FLOAT32:
      return circle::FullyConnectedOptionsWeightsFormat_SHUFFLED16x1FLOAT32;
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::WeightsFormat", oops::to_uint32(format));
  }
}

circle::DimensionType to_circle_dimensiontype(luci::DimensionType type)
{
  switch (type)
  {
    case luci::DimensionType::DENSE:
      return circle::DimensionType_DENSE;
    case luci::DimensionType::SPARSE_CSR:
      return circle::DimensionType_SPARSE_CSR;
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::DimensionType", oops::to_uint32(type));
  }
}

flatbuffers::Offset<void> to_circle_sparse_index_vector(flatbuffers::FlatBufferBuilder &fb,
                                                        const SparseIndexVector &sparse_idx_vec)
{
  auto type = sparse_idx_vec.type();
  switch (type)
  {
    case luci::SparseIndexVectorType::NONE:
      return flatbuffers::Offset<void>();
    case luci::SparseIndexVectorType::I32:
    {
      return circle::CreateInt32VectorDirect(fb, sparse_idx_vec.as_int32_vector()).Union();
    }
    case luci::SparseIndexVectorType::U16:
    {
      return circle::CreateUint16VectorDirect(fb, sparse_idx_vec.as_uint16_vector()).Union();
    }
    case luci::SparseIndexVectorType::U8:
    {
      return circle::CreateUint8VectorDirect(fb, sparse_idx_vec.as_uint8_vector()).Union();
    }
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::SparseIndexVectorType",
                     oops::to_uint32(type));
  }
}

circle::SparseIndexVector to_circle_sparse_index_vector_type(luci::SparseIndexVectorType type)
{
  switch (type)
  {
    case luci::SparseIndexVectorType::NONE:
      return circle::SparseIndexVector_NONE;
    case luci::SparseIndexVectorType::I32:
      return circle::SparseIndexVector_Int32Vector;
    case luci::SparseIndexVectorType::U16:
      return circle::SparseIndexVector_Uint16Vector;
    case luci::SparseIndexVectorType::U8:
      return circle::SparseIndexVector_Uint8Vector;
    default:
      INTERNAL_EXN_V("trying to convert unsupported luci::SparseIndexVectorType",
                     oops::to_uint32(type));
  }
}

circle::BuiltinOperator circle_builtin_operator(const luci::CircleNode *node)
{
  return node->accept(&BuiltinOperatorMappingRule::get());
}

circle::BuiltinOptions circle_builtin_options(const luci::CircleNode *node)
{
  if (auto cast = dynamic_cast<const luci::CircleCast *>(node))
  {
    return (cast->out_data_type() == loco::DataType::Unknown) ? circle::BuiltinOptions_NONE
                                                              : circle::BuiltinOptions_CastOptions;
  }

  return node->accept(&BuiltinOptionsMappingRule::get());
}

std::string circle_custom_code(const luci::CircleNode *node)
{
  if (auto custom_node = dynamic_cast<const luci::CircleCustom *>(node))
  {
    return custom_node->custom_code();
  }

  return "";
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
circle_custom_options(flatbuffers::FlatBufferBuilder &fb, const luci::CircleNode *node)
{
  if (auto custom_node = dynamic_cast<const luci::CircleCustom *>(node))
  {
    std::vector<uint8_t> custom_options_vec{custom_node->custom_options().begin(),
                                            custom_node->custom_options().end()};
    return fb.CreateVector(custom_options_vec);
  }

  return 0;
}

} // namespace luci

namespace luci
{

uint32_t SerializedModelData::registerBuiltinOpcode(circle::BuiltinOperator builtin_code,
                                                    const std::string &custom_code,
                                                    const int32_t op_version)
{
  assert(op_version > 0);

  auto it = _operator_codes.find(OpCode{builtin_code, custom_code, op_version});
  if (it != _operator_codes.end())
  {
    return it->second;
  }
  auto idx = static_cast<uint32_t>(_operator_codes.size());
  _operator_codes.emplace(OpCode{builtin_code, custom_code, op_version}, idx);
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
