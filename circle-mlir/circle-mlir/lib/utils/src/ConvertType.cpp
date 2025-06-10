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

// from tensorflow/compiler/mlir/lite/utils/convert_type.cc

#include "circle-mlir/utils/ConvertType.h"

#include <circle_schema/schema_generated.h>

#include <mlir/IR/Builders.h>     // from @llvm-project
#include <mlir/IR/BuiltinTypes.h> // from @llvm-project
#include <mlir/IR/Types.h>        // from @llvm-project

#include <stdexcept>

namespace circle
{

circle::TensorType ConvertTypeToTensorType(mlir::Type type)
{
  if (type.isF16())
  {
    return circle::TensorType_FLOAT16;
  }
  else if (type.isF32())
  {
    return circle::TensorType_FLOAT32;
  }
  else if (type.isF64())
  {
    return circle::TensorType_FLOAT64;
  }
  /* TODO support this when needed
  else if (type.isa<mlir::TF::StringType>())
  {
    return circle::TensorType_STRING;
  }
  */
  else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type))
  {
    if (complex_type.getElementType().isF32())
    {
      return circle::TensorType_COMPLEX64;
    }
    else if (complex_type.getElementType().isF64())
    {
      return circle::TensorType_COMPLEX128;
    }
    throw std::runtime_error("invalid complex Type in conversion");
  }
  else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type))
  {
    switch (itype.getWidth())
    {
      case 1:
        return circle::TensorType_BOOL;
      case 4:
        if (itype.isUnsigned())
          return circle::TensorType_UINT4;
        else
          return circle::TensorType_INT4;
      case 8:
        if (itype.isUnsigned())
          return circle::TensorType_UINT8;
        else
          return circle::TensorType_INT8;
      case 16:
        return circle::TensorType_INT16;
      case 32:
        return circle::TensorType_INT32;
      case 64:
        if (itype.isUnsigned())
          return circle::TensorType_UINT64;
        else
          return circle::TensorType_INT64;
      default:
        throw std::runtime_error("invalid integer Type in conversion");
    }
  }
  throw std::runtime_error("invalid Type in conversion");
}

mlir::Type ConvertElementType(circle::TensorType type, mlir::Builder builder)
{
  switch (type)
  {
    case circle::TensorType_FLOAT16:
      return builder.getF16Type();
    case circle::TensorType_FLOAT32:
      return builder.getF32Type();
    case circle::TensorType_FLOAT64:
      return builder.getF64Type();
    case circle::TensorType_INT32:
      return builder.getIntegerType(32);
    case circle::TensorType_UINT16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case circle::TensorType_UINT32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case circle::TensorType_UINT8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case circle::TensorType_UINT4:
      return builder.getIntegerType(4, /*isSigned=*/false);
    case circle::TensorType_INT64:
      return builder.getIntegerType(64);
    /*
    TODO support this when needed
    case circle::TensorType_STRING:
      return mlir::TF::StringType::get(builder.getContext());
    */
    case circle::TensorType_BOOL:
      return builder.getI1Type();
    case circle::TensorType_INT16:
      return builder.getIntegerType(16);
    case circle::TensorType_COMPLEX64:
      return mlir::ComplexType::get(builder.getF32Type());
    case circle::TensorType_COMPLEX128:
      return mlir::ComplexType::get(builder.getF64Type());
    case circle::TensorType_INT4:
      return builder.getIntegerType(4);
    case circle::TensorType_INT8:
      return builder.getIntegerType(8);
    case circle::TensorType_UINT64:
      return builder.getIntegerType(64, /*isSigned=*/false);
      /*
      TODO support this when needed
      case circle::TensorType_RESOURCE:
        return mlir::TF::ResourceType::get(builder.getContext());
      case circle::TensorType_VARIANT:
        return mlir::TF::VariantType::get(builder.getContext());
      */
  }
  throw std::runtime_error("invalid type in conversion");
}

} // namespace circle
