/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/flatbuffer_operator.cc

#include "circle-mlir/utils/FlatbufferOperator.h"
#include "circle-mlir/utils/ConvertType.h"

#include <circle_schema/schema_generated.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>

#include <utility> // used in OperatorConverters.inc
#include <vector>  // used in OperatorConverters.inc

// TODO(jpienaar): This is a placeholder. This should be done in more efficient
// way when part of the translation of module.
static circle::ActivationFunctionType
ConvertCIR_AFAttrForOptionWriter(llvm::StringRef str, flatbuffers::FlatBufferBuilder *builder)
{
  return llvm::StringSwitch<circle::ActivationFunctionType>(str)
    .Case("NONE", circle::ActivationFunctionType_NONE)
    .Case("RELU", circle::ActivationFunctionType_RELU)
    .Case("RELU_N1_TO_1", circle::ActivationFunctionType_RELU_N1_TO_1)
    .Case("RELU6", circle::ActivationFunctionType_RELU6)
    .Case("TANH", circle::ActivationFunctionType_TANH)
    .Case("SIGN_BIT", circle::ActivationFunctionType_SIGN_BIT);
}

static circle::FullyConnectedOptionsWeightsFormat
ConvertCIR_FullyConnectedOptionsWeightFormatAttrForOptionWriter(
  llvm::StringRef str, flatbuffers::FlatBufferBuilder *builder)
{
  return llvm::StringSwitch<circle::FullyConnectedOptionsWeightsFormat>(str)
    .Case("DEFAULT", circle::FullyConnectedOptionsWeightsFormat_DEFAULT)
    .Case("SHUFFLED4x16INT8", circle::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
}

static circle::Padding
ConvertCIR_PaddingAttrForOptionWriter(llvm::StringRef str, flatbuffers::FlatBufferBuilder *builder)
{
  return llvm::StringSwitch<circle::Padding>(str)
    .Case("SAME", circle::Padding_SAME)
    .Case("VALID", circle::Padding_VALID);
}

// NOTE this returns type if type is INT64 or INT32
//      not sure why implementation is like this in TensorFlow
static circle::TensorType
ConvertDerivedCircleTypeAttrForOptionWriter(circle::TensorType type,
                                            flatbuffers::FlatBufferBuilder *builder)
{
  if (type == circle::TensorType_INT64)
  {
    return circle::TensorType_INT64;
  }
  else if (type == circle::TensorType_INT32)
  {
    return circle::TensorType_INT32;
  }
  llvm_unreachable("invalid type in conversion.");
}

static circle::TensorType
ConvertDerivedTypeAttrForOptionWriter(mlir::Type type, flatbuffers::FlatBufferBuilder *builder)
{
  return circle::ConvertTypeToTensorType(type);
}

// BoolAttr already returns a bool as required by flatbuffer builders.
static bool ConvertBoolAttrForOptionWriter(bool b, flatbuffers::FlatBufferBuilder *builder)
{
  return b;
}

// Overloading of ConvertBoolAttrForOptionWriter which takes Optional<bool> as
// an input. If value is not specified, false is set for the attribute.
static bool ConvertBoolAttrForOptionWriter(std::optional<bool> b,
                                           flatbuffers::FlatBufferBuilder *builder)
{
  return b.has_value() ? b.value() : false;
}

// I32Attr already returns an int as required by flatbuffer builders.
static int ConvertI32AttrForOptionWriter(int i, flatbuffers::FlatBufferBuilder *builder)
{
  return i;
}

static flatbuffers::Offset<flatbuffers::Vector<int32_t>>
ConvertI64ArrayAttrForOptionWriter(mlir::ArrayAttr attrArray,
                                   flatbuffers::FlatBufferBuilder *builder)
{
  std::vector<int32_t> intVec;
  intVec.reserve(attrArray.getValue().size());
  for (auto attr : attrArray.getValue())
  {
    intVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
  }
  return builder->CreateVector(intVec);
}

// F32Attr already returns a float as required by flatbuffer builders.
static float ConvertF32AttrForOptionWriter(llvm::APFloat f, flatbuffers::FlatBufferBuilder *builder)
{
  return f.convertToFloat();
}

static mlir::Attribute BuildBoolAttr(bool value, mlir::Builder builder)
{
  return builder.getBoolAttr(value);
}

static mlir::Attribute BuildF32Attr(float value, mlir::Builder builder)
{
  return builder.getF32FloatAttr(value);
}

static mlir::Attribute BuildI32Attr(int32_t value, mlir::Builder builder)
{
  return builder.getI32IntegerAttr(value);
}

static mlir::Attribute BuildI64ArrayAttr(std::vector<int32_t> value, mlir::Builder builder)
{
  std::vector<int64_t> typecast(value.begin(), value.end());
  return builder.getI64ArrayAttr(typecast);
}

static mlir::Attribute BuildCIR_AFAttr(circle::ActivationFunctionType value, mlir::Builder builder)
{
  const char *option_name = circle::EnumNameActivationFunctionType(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute
BuildCIR_FullyConnectedOptionsWeightFormatAttr(circle::FullyConnectedOptionsWeightsFormat value,
                                               mlir::Builder builder)
{
  const char *option_name = circle::EnumNameFullyConnectedOptionsWeightsFormat(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildCIR_PaddingAttr(circle::Padding value, mlir::Builder builder)
{
  const char *option_name = circle::EnumNamePadding(value);
  return builder.getStringAttr(option_name);
}

#include <circle-mlir/dialect/CircleDialect.h>

static circle::MirrorPadMode
ConvertCIR_MirrorPaddingAttrForOptionWriter(mlir::Circle::MirrorPaddingType padding,
                                            flatbuffers::FlatBufferBuilder *builder)
{
  switch (padding)
  {
    case mlir::Circle::MirrorPaddingType::REFLECT:
      return circle::MirrorPadMode_REFLECT;
    case mlir::Circle::MirrorPaddingType::SYMMETRIC:
      return circle::MirrorPadMode_SYMMETRIC;
  }
  llvm_unreachable("invalid mirror_pad_enum in conversion.");
}

static mlir::Attribute BuildCIR_MirrorPaddingAttr(circle::MirrorPadMode value,
                                                  mlir::Builder builder)
{
  mlir::Circle::MirrorPaddingType padding;
  switch (value)
  {
    case circle::MirrorPadMode_REFLECT:
      padding = mlir::Circle::MirrorPaddingType::REFLECT;
      break;
    case circle::MirrorPadMode_SYMMETRIC:
    default:
      padding = mlir::Circle::MirrorPaddingType::SYMMETRIC;
      break;
  }
  return mlir::Circle::MirrorPaddingTypeAttr::get(builder.getContext(), padding);
}

// Pull in FlatBuffer writers for Circle generated using TableGen
#include "mlir/OperatorConverters.inc"
