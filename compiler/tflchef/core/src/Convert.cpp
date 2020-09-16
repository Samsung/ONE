/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Convert.h"

#include <stdexcept>

tflite::Padding as_tflite_padding(const tflchef::Padding &value)
{
  switch (value)
  {
    case tflchef::SAME:
      return tflite::Padding_SAME;
    case tflchef::VALID:
      return tflite::Padding_VALID;
    default:
      break;
  }

  throw std::runtime_error{"Unknown padding value"};
}

tflite::ActivationFunctionType as_tflite_activation(const tflchef::Activation &value)
{
  switch (value)
  {
    case tflchef::NONE:
      return tflite::ActivationFunctionType_NONE;
    case tflchef::RELU:
      return tflite::ActivationFunctionType_RELU;
    case tflchef::RELU_N1_TO_1:
      return tflite::ActivationFunctionType_RELU_N1_TO_1;
    case tflchef::RELU6:
      return tflite::ActivationFunctionType_RELU6;
    case tflchef::TANH:
      return tflite::ActivationFunctionType_TANH;
    case tflchef::SIGN_BIT:
      return tflite::ActivationFunctionType_SIGN_BIT;
    default:
      break;
  }

  throw std::runtime_error{"Unknown activation"};
}

tflite::TensorType as_tflite_tensortype(const tflchef::TensorType &value)
{
  switch (value)
  {
    case tflchef::FLOAT32:
      return tflite::TensorType_FLOAT32;
    case tflchef::INT32:
      return tflite::TensorType_INT32;
    case tflchef::UINT8:
      return tflite::TensorType_UINT8;
    case tflchef::INT64:
      return tflite::TensorType_INT64;
    case tflchef::BOOL:
      return tflite::TensorType_BOOL;
    default:
      break;
  }

  throw std::runtime_error{"Unknown tensor type"};
}

tflite::MirrorPadMode as_tflite_mirrorpadmode(const tflchef::MirrorPadMode &value)
{
  switch (value)
  {
    case tflchef::REFLECT:
      return tflite::MirrorPadMode_REFLECT;
    case tflchef::SYMMETRIC:
      return tflite::MirrorPadMode_SYMMETRIC;
    default:
      break;
  }

  throw std::runtime_error{"Unknown mirrorpad mode"};
}

tflite::DimensionType as_tflite_dimensiontype(const tflchef::DimensionType &value)
{
  switch (value)
  {
    case tflchef::DimensionType::DENSE:
      return tflite::DimensionType_DENSE;
    case tflchef::DimensionType::SPARSE_CSR:
      return tflite::DimensionType_SPARSE_CSR;
    default:
      break;
  }

  throw std::runtime_error("Unknown dimension type");
}

tflite::SparseIndexVector as_tflite_sparse_idx_vec_type(const tflchef::SparseIndexVecType &value)
{
  switch (value)
  {
    case tflchef::SparseIndexVecType::SparseIdxVecType_NONE:
      return tflite::SparseIndexVector_NONE;
    case tflchef::SparseIndexVecType::INT32VEC:
      return tflite::SparseIndexVector_Int32Vector;
    case tflchef::SparseIndexVecType::UINT16VEC:
      return tflite::SparseIndexVector_Uint16Vector;
    case tflchef::SparseIndexVecType::UINT8VEC:
      return tflite::SparseIndexVector_Uint8Vector;
    default:
      break;
  }

  throw std::runtime_error("Unknown SparseIndexVector type");
}

flatbuffers::Offset<void>
as_tflite_sparse_index_vec(flatbuffers::FlatBufferBuilder &fb,
                           const ::tflchef::TensorSparsity_IndexVec &value)
{
  auto sparse_idx_type = value.type();

  switch (sparse_idx_type)
  {
    case tflchef::SparseIndexVecType::SparseIdxVecType_NONE:
      return flatbuffers::Offset<void>();
    case tflchef::SparseIndexVecType::INT32VEC:
    {
      auto values_vec_int32 = std::vector<int32_t>{value.dim().begin(), value.dim().end()};
      auto values_int32 = fb.CreateVector(values_vec_int32);
      return tflite::CreateInt32Vector(fb, values_int32).Union();
    }
    case tflchef::SparseIndexVecType::UINT16VEC:
    {
      auto values_vec_uint16 = std::vector<uint16_t>{value.dim().begin(), value.dim().end()};
      auto values_uint16 = fb.CreateVector(values_vec_uint16);
      return tflite::CreateUint16Vector(fb, values_uint16).Union();
    }
    case tflchef::SparseIndexVecType::UINT8VEC:
    {
      auto values_vec_uint8 = std::vector<uint8_t>{value.dim().begin(), value.dim().end()};
      auto values_uint8 = fb.CreateVector(values_vec_uint8);
      return tflite::CreateUint8Vector(fb, values_uint8).Union();
    }
    default:
      break;
  }

  throw std::runtime_error("Unknown SparseIndexVector type");
}
