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
      throw std::runtime_error("tflite2circle: wrong op");
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

} // namespace tflite2circle
