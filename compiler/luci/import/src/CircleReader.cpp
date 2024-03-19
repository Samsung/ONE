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

#include <mio_circle/Helper.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

namespace luci
{

const char *tensor_name(const circle::Tensor *tensor)
{
  assert(tensor != nullptr);

  if (tensor->name() == nullptr || std::string(tensor->name()->c_str()).empty())
    return "(noname)";

  return tensor->name()->c_str();
}

const circle::QuantizationParameters *tensor_quantization(const circle::Tensor *tensor)
{
  assert(tensor != nullptr);
  return tensor->quantization();
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
    case circle::TensorType_UINT4:
      return loco::DataType::U4;
    case circle::TensorType_INT64:
      return loco::DataType::S64;
    case circle::TensorType_STRING:
      return loco::DataType::STRING;
    case circle::TensorType_BOOL:
      return loco::DataType::BOOL;
    case circle::TensorType_INT16:
      return loco::DataType::S16;
    case circle::TensorType_COMPLEX64:
      break;
    case circle::TensorType_INT8:
      return loco::DataType::S8;
    case circle::TensorType_INT4:
      return loco::DataType::S4;
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

std::unique_ptr<CircleQuantParam> luci_quantparam(const circle::QuantizationParameters *qparams)
{
  // create temporary unpacked API object
  assert(qparams != nullptr);
  circle::QuantizationParametersT quantization;
  qparams->UnPackTo(&quantization);

  return luci_quantparam(&quantization);
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

std::unique_ptr<SparsityParam> luci_sparsityparam(const circle::SparsityParameters *sparparam)
{
  // create temporary unpacked API object
  assert(sparparam != nullptr);
  circle::SparsityParametersT sparsity;
  sparparam->UnPackTo(&sparsity);

  return luci_sparsityparam(&sparsity);
}

void copy_tensor_attributes(const circle::Tensor *tensor, CircleNode *node)
{
  assert(tensor != nullptr);

  node->name(tensor_name(tensor));
  node->dtype(luci_datatype(tensor->type()));

  const auto tensor_shape_signature = wrap(tensor->shape_signature());
  const auto tensor_shape = wrap(tensor->shape());
  assert(tensor_shape_signature.size() == 0 ||
         tensor_shape_signature.size() == tensor_shape.size());

  const auto dims = tensor_shape; // in NHWC
  node->rank(dims.size());
  for (uint32_t r = 0; r < dims.size(); ++r)
  {
    if (tensor_shape_signature.size() > 0 && tensor_shape_signature.at(r) == -1)
      node->dim(r).unset();
    else
      node->dim(r).set(dims[r]);
  }

  const auto quantization = tensor->quantization();
  if (quantization != nullptr)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam)
      node->quantparam(std::move(quantparam));
  }

  const auto sparsity = tensor->sparsity();
  if (sparsity != nullptr)
  {
    auto sparsityparam = luci_sparsityparam(sparsity);
    if (sparsityparam)
      node->sparsityparam(std::move(sparsityparam));
  }
}

std::string fb_string2std_string(const flatbuffers::String *fb_str)
{
  return fb_str == nullptr ? "" : fb_str->str();
}

circle::BuiltinOperator CircleReader::builtin_code(const circle::Operator *op) const
{
  assert(op != nullptr);

  const auto op_codes = opcodes();
  uint32_t index = op->opcode_index();
  assert(index < op_codes.size());
  const auto opcode = op_codes[index];
  assert(opcode != nullptr);

  return mio::circle::builtin_code_neutral(opcode);
}

std::string CircleReader::opcode_name(const circle::Operator *op) const
{
  assert(op != nullptr);

  const auto op_codes = opcodes();
  uint32_t index = op->opcode_index();
  assert(index < op_codes.size());
  const auto opcode = op_codes[index];

  return mio::circle::opcode_name(opcode);
}

bool CircleReader::parse(const circle::Model *model)
{
  assert(model != nullptr);

  // for direct pointer access
  _model = model;

  return true;
}

bool CircleReader::select_subgraph(uint32_t sgindex)
{
  if (num_subgraph() <= sgindex)
  {
    assert(false);
    return false;
  }

  // for direct pointer access
  auto subgraphs = _model->subgraphs();
  assert(subgraphs != nullptr);

  _current_subgraph = subgraphs->Get(sgindex);
  assert(_current_subgraph != nullptr);

  return true;
}

template <typename T>
VectorWrapper<T>::VectorWrapper(const flatbuffers::Vector<T> *ptr) : _vector(ptr)
{
  // Do nothing
}

template <typename T> uint32_t VectorWrapper<T>::size() const
{
  return null() ? 0 : _vector->size();
}

template <typename T> const T *VectorWrapper<T>::data() const
{
  return null() ? nullptr : _vector->data();
}

template <typename T> typename VectorWrapper<T>::iterator VectorWrapper<T>::begin() const
{
  return null() ? iterator(nullptr, 0) : _vector->begin();
}

template <typename T> typename VectorWrapper<T>::iterator VectorWrapper<T>::end() const
{
  return null() ? begin() : _vector->end();
}

template <typename T> typename VectorWrapper<T>::value_type VectorWrapper<T>::at(uint32_t i) const
{
  if (i >= size())
  {
    // TODO find better error message
    throw std::range_error("Access to prohibited vector element");
  }

  return _vector->Get(i);
}

template <typename T>
typename VectorWrapper<T>::value_type VectorWrapper<T>::operator[](uint32_t i) const
{
  return at(i);
}

template <typename T> bool VectorWrapper<T>::null() const { return _vector == nullptr; }
template <typename T> bool VectorWrapper<T>::empty() const { return size() == 0; }

#define REGISTER_WRAPPER(T) template class VectorWrapper<T>
REGISTER_WRAPPER(flatbuffers::Offset<circle::SubGraph>);
REGISTER_WRAPPER(flatbuffers::Offset<circle::Buffer>);
REGISTER_WRAPPER(flatbuffers::Offset<circle::Tensor>);
REGISTER_WRAPPER(flatbuffers::Offset<circle::Operator>);
REGISTER_WRAPPER(flatbuffers::Offset<circle::OperatorCode>);
REGISTER_WRAPPER(flatbuffers::Offset<circle::Metadata>);
REGISTER_WRAPPER(int32_t);
REGISTER_WRAPPER(uint8_t);
#undef REGISTER_WRAPPER

} // namespace luci
