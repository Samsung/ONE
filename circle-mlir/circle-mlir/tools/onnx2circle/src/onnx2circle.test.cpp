/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>

#include <string>

// delcare methods of onnx2circle.cpp to test
namespace onnx2circle
{

int loadONNX(const std::string &onnx_path, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module);

} // namespace onnx2circle

#include <gtest/gtest.h>

TEST(LoadONNXTest, NonExistFile_NEG)
{
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  std::string invalid_filename = "/no_such_folder/no_such_file_in_storage.mlir";
  auto result = onnx2circle::loadONNX(invalid_filename, context, module);
  ASSERT_NE(0, result);

  invalid_filename = "/no_such_folder/no_such_file_in_storage.onnx";
  result = onnx2circle::loadONNX(invalid_filename, context, module);
  ASSERT_NE(0, result);
}

TEST(LoadONNXTest, NotSupportedExtension_NEG)
{
  std::string invalid_filename = "somefile.blabla";

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  auto result = onnx2circle::loadONNX(invalid_filename, context, module);
  ASSERT_NE(0, result);
}
