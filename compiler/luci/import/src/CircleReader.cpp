/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/CircleReader.h"

#include <memory>
#include <sstream>
#include <string>

namespace luci
{

bool is_valid(const circle::OperatorCodeT &opcode)
{
  circle::BuiltinOperator code = opcode.builtin_code;
  return (circle::BuiltinOperator_MIN <= code && code <= circle::BuiltinOperator_MAX);
}

bool is_custom(const circle::OperatorCodeT &opcode)
{
  circle::BuiltinOperator code = opcode.builtin_code;
  return (code == circle::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const circle::OperatorCodeT &opcode)
{
  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid)";
    return oss.str();
  }

  if (is_custom(opcode))
  {
    if (opcode.custom_code.empty())
      return "(invalid custom)";

    return opcode.custom_code;
  }

  circle::BuiltinOperator code = opcode.builtin_code;
  return circle::EnumNameBuiltinOperator(code);
}

const char *tensor_name(const circle::TensorT &tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  if (!tensor.name.empty())
    return tensor.name.c_str();

  return kEmptyTensorName;
}

const circle::QuantizationParametersT *tensor_quantization(const circle::TensorT &tensor)
{
  return tensor.quantization.get();
}

loco::DataType luci_datatype(const circle::TensorType type)
{
  switch (type)
  {
    case circle::TensorType_FLOAT32:
      return loco::DataType::FLOAT32;
    case circle::TensorType_FLOAT16:
      return loco::DataType::FLOAT16;
    case circle::TensorType_INT32:
      return loco::DataType::S32;
    case circle::TensorType_UINT8:
      return loco::DataType::U8;
    case circle::TensorType_INT64:
      return loco::DataType::S64;
    case circle::TensorType_STRING:
      break;
    case circle::TensorType_BOOL:
      return loco::DataType::BOOL;
    case circle::TensorType_INT16:
      return loco::DataType::S16;
    case circle::TensorType_COMPLEX64:
      break;
    case circle::TensorType_INT8:
      return loco::DataType::S8;
    default:
      break;
  }
  assert(false);
  return loco::DataType::Unknown;
}

FusedActFunc luci_actfunc(const circle::ActivationFunctionType type)
{
  switch (type)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
      return luci::FusedActFunc::NONE;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      return luci::FusedActFunc::RELU;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      return luci::FusedActFunc::RELU_N1_TO_1;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      return luci::FusedActFunc::RELU6;
    case circle::ActivationFunctionType::ActivationFunctionType_TANH:
      return luci::FusedActFunc::TANH;
    case circle::ActivationFunctionType::ActivationFunctionType_SIGN_BIT:
      return luci::FusedActFunc::SIGN_BIT;
    default:
      break;
  }
  assert(false);
  return luci::FusedActFunc::UNDEFINED;
}

Padding luci_padding(const circle::Padding padding)
{
  switch (padding)
  {
    case circle::Padding::Padding_SAME:
      return Padding::SAME;
    case circle::Padding::Padding_VALID:
      return Padding::VALID;
  }
  assert(false);
  return Padding::UNDEFINED;
}

MirrorPadMode luci_mirrorpad_mode(const circle::MirrorPadMode mode)
{
  switch (mode)
  {
    case circle::MirrorPadMode::MirrorPadMode_REFLECT:
      return MirrorPadMode::REFLECT;
    case circle::MirrorPadMode::MirrorPadMode_SYMMETRIC:
      return MirrorPadMode::SYMMETRIC;
  }
  assert(false);
  return MirrorPadMode::UNDEFINED;
}

luci::CircleFullyConnected::WeightsFormat
luci_weights_format(const circle::FullyConnectedOptionsWeightsFormat weights_format)
{
  switch (weights_format)
  {
    case circle::FullyConnectedOptionsWeightsFormat_DEFAULT:
      return luci::CircleFullyConnected::WeightsFormat::DEFAULT;
    case circle::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
      return luci::CircleFullyConnected::WeightsFormat::SHUFFLED4x16INT8;
    case circle::FullyConnectedOptionsWeightsFormat_SHUFFLED16x1FLOAT32:
      return luci::CircleFullyConnected::WeightsFormat::SHUFFLED16x1FLOAT32;
    default:
      throw std::runtime_error("Invalid FullyConnectedOptionsWeightsFormat");
  }
}

DimensionType luci_dim_type(const circle::DimensionType dim_type)
{
  switch (dim_type)
  {
    case circle::DimensionType_DENSE:
      return DimensionType::DENSE;
    case circle::DimensionType_SPARSE_CSR:
      return DimensionType::SPARSE_CSR;
    default:
      throw std::runtime_error("Invalid DimensionType");
  }
}

SparseIndexVector
luci_sparse_index_vector(const circle::SparseIndexVectorUnion &sparse_index_vector)
{
  switch (sparse_index_vector.type)
  {
    case circle::SparseIndexVector_NONE:
      return SparseIndexVector{SparseIndexVectorType::NONE, nullptr};
    case circle::SparseIndexVector_Int32Vector:
    {
      const auto const_vec_ptr =
        static_cast<const void *>(&(sparse_index_vector.AsInt32Vector()->values));
      return SparseIndexVector{SparseIndexVectorType::I32, const_vec_ptr};
    }
    case circle::SparseIndexVector_Uint16Vector:
    {
      const auto const_vec_ptr =
        static_cast<const void *>(&(sparse_index_vector.AsUint16Vector()->values));
      return SparseIndexVector{SparseIndexVectorType::U16, const_vec_ptr};
    }
    case circle::SparseIndexVector_Uint8Vector:
    {
      const auto const_vec_ptr =
        static_cast<const void *>(&(sparse_index_vector.AsUint8Vector()->values));
      return SparseIndexVector{SparseIndexVectorType::U8, const_vec_ptr};
    }
    default:
      throw std::runtime_error("Invalid SparseIndexVector type");
  }
}

std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParametersT *quantization)
{
  const auto &min = quantization->min;
  const auto &max = quantization->max;
  const auto &scale = quantization->scale;
  const auto &zero_point = quantization->zero_point;
  const auto &quantized_dimension = quantization->quantized_dimension;

  if ((!min.empty() && !max.empty()) || (!scale.empty() && !zero_point.empty()))
  {
    auto quantparam = std::make_unique<CircleQuantParam>();

    quantparam->min = min;
    quantparam->max = max;
    quantparam->scale = scale;
    quantparam->zerop = zero_point;
    quantparam->quantized_dimension = quantized_dimension;

    return quantparam;
  }

  return nullptr;
}

std::unique_ptr<SparsityParam> luci_sparsityparam(const circle::SparsityParametersT *sparsity)
{
  assert(sparsity);
  const auto &traversal_order = sparsity->traversal_order;
  const auto &block_map = sparsity->block_map;
  const auto &dim_metadata = sparsity->dim_metadata;

  // TODO find a condition that should return nullptr
  auto sparsityparam = std::make_unique<SparsityParam>();

  sparsityparam->traversal_order = traversal_order;
  sparsityparam->block_map = block_map;
  for (const auto &dm : dim_metadata)
  {
    sparsityparam->dim_metadata.emplace_back(luci_dim_type(dm->format), dm->dense_size,
                                             luci_sparse_index_vector(dm->array_segments),
                                             luci_sparse_index_vector(dm->array_indices));
  }

  return sparsityparam;
}

void copy_tensor_attributes(const circle::TensorT &tensor, CircleNode *node)
{
  node->name(tensor_name(tensor));
  node->dtype(luci_datatype(tensor.type));

  std::vector<int32_t> dims = tensor.shape; // in NHWC
  node->rank(dims.size());
  for (uint32_t r = 0; r < dims.size(); ++r)
  {
    if (tensor.shape_signature.size() > 0 && tensor.shape_signature.at(r) == -1)
      node->dim(r).unset();
    else
      node->dim(r).set(dims[r]);
  }

  const auto *quantization = tensor.quantization.get();
  if (quantization != nullptr)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam)
      node->quantparam(std::move(quantparam));
  }

  const auto *sparsity = tensor.sparsity.get();
  if (sparsity != nullptr)
  {
    auto sparsityparam = luci_sparsityparam(sparsity);
    if (sparsityparam)
      node->sparsityparam(std::move(sparsityparam));
  }
}

circle::BuiltinOperator CircleReader::builtin_code(const circle::OperatorT &op) const
{
  const auto &op_codes = opcodes();
  uint32_t index = op.opcode_index;
  assert(index < op_codes.size());
  const circle::OperatorCodeT &opcode = *op_codes[index];

  return opcode.builtin_code;
}

std::string CircleReader::opcode_name(const circle::OperatorT &op) const
{
  const auto &op_codes = opcodes();
  uint32_t index = op.opcode_index;
  assert(index < op_codes.size());
  const circle::OperatorCodeT &opcode = *op_codes[index];

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  return ::luci::opcode_name(opcode);
}

bool CircleReader::parse(const circle::Model *model)
{
  assert(model != nullptr);

  _model.reset(model->UnPack());

  // for direct pointer access
  _model_ptr = model;

  return true;
}

bool CircleReader::select_subgraph(uint32_t sgindex)
{
  if (_model->subgraphs.size() <= sgindex)
  {
    assert(false);
    return false;
  }

  _current_subgraph = _model->subgraphs[sgindex].get();

  // for direct pointer access
  auto subgraphs = _model_ptr->subgraphs();
  const circle::SubGraph *subgraph = (*subgraphs)[sgindex];

  _tensors_ptr = subgraph->tensors();

  return true;
}

} // namespace luci
