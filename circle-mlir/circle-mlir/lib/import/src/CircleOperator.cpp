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

#include "CircleOperator.h"

#include <circle-mlir/dialect/CircleDialect.h>

#include <llvm/ADT/Twine.h>
#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <cassert>

namespace circle
{

// from tensorflow/lite/schema/schema_utils.cc

BuiltinOperator GetBuiltinCode(const OperatorCodeT *op_code)
{
  // Caller should guarantee that the given argument value is not a nullptr.
  assert(op_code != nullptr);

  if (op_code->builtin_code >= 0)
    return std::max(op_code->builtin_code,
                    static_cast<BuiltinOperator>(op_code->deprecated_builtin_code));
  // circle extended
  return op_code->builtin_code;
}

} // namespace circle

namespace mlir
{

bool IsStablehloOp(const circle::OperatorCodeT &op_code)
{
  // NOTE As StableHLO is not treated yet, we can just return false
  llvm::StringRef op_name(circle::EnumNameBuiltinOperator(circle::GetBuiltinCode(&op_code)));
  return op_name.starts_with("STABLEHLO_");
}

std::string GetMlirOpNameFromOpCode(const circle::OperatorCodeT &op_code)
{
  auto builtin_code = circle::GetBuiltinCode(&op_code);
  if (builtin_code == circle::BuiltinOperator_IF)
  {
    // TODO handle 'if' when approached here
    assert(false);
    return std::string("Circle.If");
  }

  llvm::StringRef op_name(circle::EnumNameBuiltinOperator(builtin_code));

  // If the Op name contains stablehlo
  if (IsStablehloOp(op_code))
  {
    // TODO handle 'if' when approached here
    assert(false);
    return llvm::Twine("stablehlo.", op_name.drop_front(10).lower()).str();
  }
  return llvm::Twine("Circle.", op_name.lower()).str();
}

bool CustomOptionsToAttributes(const std::string &custom_code,
                               const std::vector<uint8_t> &custom_options, mlir::Builder builder,
                               mlir::Location loc,
                               llvm::SmallVectorImpl<mlir::NamedAttribute> *attributes)
{
  attributes->emplace_back(builder.getNamedAttr("custom_code", builder.getStringAttr(custom_code)));
  std::string content;
  content.assign(reinterpret_cast<const char *>(custom_options.data()), custom_options.size());
  attributes->emplace_back(builder.getNamedAttr(
    "custom_option", mlir::Circle::ConstBytesAttr::get(builder.getContext(), content)));

  return true;
}

} // namespace mlir
