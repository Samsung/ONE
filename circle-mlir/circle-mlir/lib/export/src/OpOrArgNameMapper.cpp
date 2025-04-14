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

// from tensorflow/compiler/mlir/op_or_arg_name_mapper.cc

#include "OpOrArgNameMapper.h"

#include "circle-mlir/dialect/NameUtils.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h" // from @llvm-project
#include "mlir/IR/Value.h"     // from @llvm-project

namespace mlir
{
namespace Circle
{

namespace
{

inline absl::string_view StringRefToView(llvm::StringRef ref)
{
  return absl::string_view(ref.data(), ref.size());
}

inline llvm::StringRef StringViewToRef(absl::string_view view)
{
  return llvm::StringRef(view.data(), view.size());
}

} // namespace

OpOrArgNameMapper::~OpOrArgNameMapper() {}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(llvm::StringRef prefix)
{
  // Insert/find if prefix is unique.
  auto prefix_it = name_to_count_.try_emplace(prefix, 0);
  if (prefix_it.second && IsUnique(prefix))
  {
    // Name is unique, increment count and return string name backed by
    // `name_to_count_`.
    ++prefix_it.first->second;
    return prefix_it.first->first();
  }

  // Add increasing number (count) to end of prefix until it is determined
  // to be unique.
  auto &val = prefix_it.first->second;
  llvm::SmallString<64> probe_name(prefix);
  probe_name.append(GetSuffixSeparator());
  const int probe_prefix_size = probe_name.size();
  while (true)
  {
    probe_name.resize(probe_prefix_size);
    // TODO(jpienaar): Subtract one so that the initial suffix is 0 instead
    // of 1.
    // TODO(jpienaar): Switch to radix 36 and update tests.
    llvm::APInt(32, val++).toString(probe_name, /*Radix=*/10, /*Signed=*/false);
    if (IsUnique(probe_name))
    {
      // Insert/find if prefix with appended number is unique.
      auto probe_name_it = name_to_count_.try_emplace(probe_name, 1);
      if (probe_name_it.second)
      {
        // Name is unique, return string name backed by `name_to_count_`.
        return probe_name_it.first->first();
      }
    }
  }
}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(OpOrVal op_or_val)
{
  auto &name = op_or_val_to_name_[op_or_val];
  if (!name.empty())
    return StringViewToRef(name);
  // Update the value in the map with unique name.
  llvm::StringRef ref = GetUniqueName(GetName(op_or_val));
  name = StringRefToView(ref);
  return ref;
}

int OpOrArgNameMapper::InitOpName(OpOrVal op_or_val, llvm::StringRef name)
{
  auto it = name_to_count_.try_emplace(name, 0);
  auto inserted = op_or_val_to_name_.try_emplace(op_or_val, StringRefToView(it.first->first()));
  (void)inserted;
  // TODO(jpienaar): Debug cases where we expect this behavior.
  // assert(inserted.second && "op_or_val already initialized");
  return it.first->second++;
}

bool OpOrArgNameMapper::IsUnique(llvm::StringRef name) { return true; }

std::string OpOrArgLocNameMapper::GetName(OpOrVal op_or_val)
{
  if (auto *op = mlir::dyn_cast<mlir::Operation *>(op_or_val))
  {
    // NOTE stop for debug version to find out if there is any Op for this case
    assert(false);
    auto name_from_loc = mlir::GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty())
      return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return std::string(op->getName().getStringRef());
  }
  auto val = mlir::dyn_cast<mlir::Value>(op_or_val);
  auto name_from_loc = mlir::GetNameFromLoc(val.getLoc());
  if (!name_from_loc.empty())
    return name_from_loc;
  // If the location is none of the expected types, then simply use name
  // generated using the op type. Follow TF convention and append the result
  // index unless 0.
  if (auto result = mlir::dyn_cast<mlir::OpResult>(val))
  {
    auto name_str = result.getOwner()->getName().getStringRef().str();
    auto value_op = val.getDefiningOp();
    if (value_op)
    {
      // if 'circle_node_name' is available, use it as the name of the op
      auto op_name = value_op->getAttrOfType<mlir::StringAttr>("circle_node_name");
      if (op_name)
        name_str = op_name.str();
    }
    if (result.getResultNumber() > 0)
      return llvm::formatv("{0}:{1}", name_str, result.getResultNumber());
    return std::string(name_str);
  }
  // Use the ASM syntax for BlockArgument
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(val))
  {
    return "arg" + std::to_string(arg.getArgNumber());
  }
  return "";
}

} // namespace Circle
} // namespace mlir
