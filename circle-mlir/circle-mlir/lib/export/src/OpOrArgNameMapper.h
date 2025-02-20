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

// from tensorflow/compiler/mlir/op_or_arg_name_mapper.h

#ifndef __CIRCLE_MLIR_EXPORT_OP_OR_ARG_NAME_MAPPER_H__
#define __CIRCLE_MLIR_EXPORT_OP_OR_ARG_NAME_MAPPER_H__

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h" // from @llvm-project
#include "mlir/IR/Value.h"     // from @llvm-project

namespace mlir
{
namespace Circle
{

// PointerUnion for operation and value.
// TODO(jpienaar): Rename the files.
using OpOrVal = llvm::PointerUnion<mlir::Operation *, mlir::Value>;

// Mapper from operation or value to name.
class OpOrArgNameMapper
{
public:
  // Returns unique name for the given prefix.
  llvm::StringRef GetUniqueName(llvm::StringRef prefix);

  // Returns unique name for the operation or value.
  llvm::StringRef GetUniqueName(OpOrVal op_or_val);

  // Initializes operation or value to map to name. Returns number of
  // operations or value already named 'name' which should be 0 else
  // GetUniqueName could return the same names for different operations or
  // values.
  // Note: Its up to the caller to decide the behavior when assigning two
  // operations or values to the same name.
  int InitOpName(OpOrVal op_or_val, llvm::StringRef name);

  virtual ~OpOrArgNameMapper();

protected:
  // Returns true if the name is unique. A derived class can override it if the
  // class maintains uniqueness in a different scope.
  virtual bool IsUnique(llvm::StringRef name);

  // Returns a constant view of the underlying map.
  const llvm::DenseMap<OpOrVal, absl::string_view> &GetMap() const { return op_or_val_to_name_; }

  // Returns the separator used before uniqueing suffix.
  virtual llvm::StringRef GetSuffixSeparator() { return ""; }

private:
  // Returns name from the location of the operation or value.
  virtual std::string GetName(OpOrVal op_or_val) = 0;

  // Maps string name to count. This map is used to help keep track of unique
  // names for operations or values.
  llvm::StringMap<int64_t> name_to_count_;
  // Maps operation or values to name. Value in map is a view of the string
  // name in `name_to_count_`. Names in `name_to_count_` are never removed.
  llvm::DenseMap<OpOrVal, absl::string_view> op_or_val_to_name_;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a name based on the location of the operation or
// value.
class OpOrArgLocNameMapper : public OpOrArgNameMapper
{
protected:
  std::string GetName(OpOrVal op_or_val) override;
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_EXPORT_OP_OR_ARG_NAME_MAPPER_H__
