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

#include "DataLookup.h"
#include "BuildBuiltinOptions.h"

namespace tflite2circle
{

circle::BuiltinOperator get_circle_builtin_code(tflite::BuiltinOperator tfl_bop)
{
  switch (tfl_bop)
  {
#define TFL_OPERATOR(OP)             \
  case tflite::BuiltinOperator_##OP: \
    return circle::BuiltinOperator_##OP;
#include "TFLOperator.lst"
#undef TFL_OPERATOR
    default:
    {
      std::string msg = "tflite2circle: Unsupported op ";
      msg = msg + tflite::EnumNameBuiltinOperator(tfl_bop);
      throw std::runtime_error(msg.c_str());
    }
  }
}

int8_t get_circle_builtin_code(int8_t tfl_bop_i8)
{
  return get_circle_builtin_code(static_cast<int32_t>(tfl_bop_i8));
}

int32_t get_circle_builtin_code(int32_t tfl_bop_i32)
{
  tflite::BuiltinOperator tfl_bop = static_cast<tflite::BuiltinOperator>(tfl_bop_i32);

  switch (tfl_bop)
  {
#define TFL_OPERATOR(OP)             \
  case tflite::BuiltinOperator_##OP: \
    return static_cast<int32_t>(circle::BuiltinOperator_##OP);
#include "TFLOperator.lst"
#undef TFL_OPERATOR
    case tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES:
      return static_cast<int32_t>(circle::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    default:
    {
      std::string msg = "tflite2circle: Unsupported op ";
      msg = msg + tflite::EnumNameBuiltinOperator(tfl_bop);
      throw std::runtime_error(msg.c_str());
    }
  }
}

circle::TensorType get_circle_tensortype(tflite::TensorType tfl_tt)
{
  switch (tfl_tt)
  {
#define TFL_TENSORTYPE(TENSORTYPE)      \
  case tflite::TensorType_##TENSORTYPE: \
    return circle::TensorType_##TENSORTYPE;
#include "TFLTensorType.lst"
#undef TFL_TENSORTYPE
    default:
      throw std::runtime_error("tflite2circle: wrong tensor type");
  }
}

circle::Padding get_circle_padding(tflite::Padding tfl_p)
{
  switch (tfl_p)
  {
    case tflite::Padding_SAME:
      return circle::Padding_SAME;
    case tflite::Padding_VALID:
      return circle::Padding_VALID;
    default:
      throw std::runtime_error("tflite2circle: wrong padding");
  }
}

circle::ActivationFunctionType
get_circle_activation_function_type(tflite::ActivationFunctionType tfl_aft)
{
  switch (tfl_aft)
  {
#define TFL_ACTIVATION_FUNCTION(TYPE)         \
  case tflite::ActivationFunctionType_##TYPE: \
    return circle::ActivationFunctionType_##TYPE;
#include "TFLActivationFunctionType.lst"
#undef TFL_ACTIVATION_FUNCTION
    default:
      throw std::runtime_error("tflite2circle: wrong activation function type.");
  }
}

flatbuffers::Offset<void> get_circle_builtin_options(flatbuffers::FlatBufferBuilder &fb,
                                                     const tflite::Operator *op)
{
  auto tflite_builtin_options_type = op->builtin_options_type();
  switch (tflite_builtin_options_type)
  {
    case tflite::BuiltinOptions_NONE:
      return flatbuffers::Offset<void>();
#define TFL_BUILTIN_OPTIONS(TYPE)     \
  case tflite::BuiltinOptions_##TYPE: \
    return build_circle_##TYPE(fb, op).Union();
#include "TFLBuiltinOptions.lst"
#undef TFL_BUILTIN_OPTIONS
    default:
      throw std::runtime_error("tflite2circle: wrong builtin options type.");
  }
}

circle::BuiltinOptions get_circle_builtin_options_type(const tflite::Operator *op)
{
  switch (op->builtin_options_type())
  {
    case tflite::BuiltinOptions_NONE:
      return circle::BuiltinOptions_NONE;
#define TFL_BUILTIN_OPTIONS(TYPE)     \
  case tflite::BuiltinOptions_##TYPE: \
    return circle::BuiltinOptions_##TYPE;
#include "TFLBuiltinOptions.lst"
#undef TFL_BUILTIN_OPTIONS
    default:
      throw std::runtime_error("tflite2circle: wrong builtin options type.");
  }
}

circle::MirrorPadMode get_circle_mirrorpad_mode(tflite::MirrorPadMode tfl_mode)
{
  switch (tfl_mode)
  {
    case tflite::MirrorPadMode_REFLECT:
      return circle::MirrorPadMode_REFLECT;
    case tflite::MirrorPadMode_SYMMETRIC:
      return circle::MirrorPadMode_SYMMETRIC;
    default:
      throw std::runtime_error("tflite2circle: wrong mirrorpad mode.");
  }
}

circle::DimensionType get_circle_dimension_type(tflite::DimensionType tfl_dim_type)
{
  switch (tfl_dim_type)
  {
    case tflite::DimensionType_DENSE:
      return circle::DimensionType_DENSE;
    case tflite::DimensionType_SPARSE_CSR:
      return circle::DimensionType_SPARSE_CSR;
    default:
      throw std::runtime_error("tflite2circle: wrong dimension type.");
  }
}

flatbuffers::Offset<void>
get_circle_sparse_index_vector(flatbuffers::FlatBufferBuilder &fb, const void *v_array,
                               const tflite::SparseIndexVector &tfl_sparse_index_vector_type)
{
  switch (tfl_sparse_index_vector_type)
  {
    case tflite::SparseIndexVector_NONE:
      return flatbuffers::Offset<void>();
    case tflite::SparseIndexVector_Int32Vector:
    {
      const tflite::Int32Vector *i32_array = static_cast<const tflite::Int32Vector *>(v_array);
      auto values_vec_int32 =
        std::vector<int32_t>{i32_array->values()->begin(), i32_array->values()->end()};
      auto values_int32 = fb.CreateVector(values_vec_int32);
      circle::Int32VectorBuilder int32_vector_builder{fb};
      int32_vector_builder.add_values(values_int32);
      return int32_vector_builder.Finish().Union();
    }
    case tflite::SparseIndexVector_Uint16Vector:
    {
      const tflite::Uint16Vector *u16_array = static_cast<const tflite::Uint16Vector *>(v_array);
      auto values_vec_uint16 =
        std::vector<uint16_t>{u16_array->values()->begin(), u16_array->values()->end()};
      auto values_uint16 = fb.CreateVector(values_vec_uint16);
      circle::Uint16VectorBuilder uint16_vector_builder{fb};
      uint16_vector_builder.add_values(values_uint16);
      return uint16_vector_builder.Finish().Union();
    }
    case tflite::SparseIndexVector_Uint8Vector:
    {
      const tflite::Uint8Vector *u8_array = static_cast<const tflite::Uint8Vector *>(v_array);
      auto values_vec_uint8 =
        std::vector<uint8_t>{u8_array->values()->begin(), u8_array->values()->end()};
      auto values_uint8 = fb.CreateVector(values_vec_uint8);
      circle::Uint8VectorBuilder uint8_vector_builder{fb};
      uint8_vector_builder.add_values(values_uint8);
      return uint8_vector_builder.Finish().Union();
    }
    default:
      throw std::runtime_error("tflite2circle: wrong SparseIndexVector type.");
  }
}

circle::SparseIndexVector
get_circle_sparse_index_vector_type(const tflite::SparseIndexVector &tfl_sparse_index_vector_type)
{
  switch (tfl_sparse_index_vector_type)
  {
    case tflite::SparseIndexVector_NONE:
      return circle::SparseIndexVector_NONE;
    case tflite::SparseIndexVector_Int32Vector:
      return circle::SparseIndexVector_Int32Vector;
    case tflite::SparseIndexVector_Uint16Vector:
      return circle::SparseIndexVector_Uint16Vector;
    case tflite::SparseIndexVector_Uint8Vector:
      return circle::SparseIndexVector_Uint8Vector;
    default:
      throw std::runtime_error("tflite2circle: wrong SparseIndexVector type.");
  }
}

} // namespace tflite2circle
