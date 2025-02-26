/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019-2022 The IBM Research Authors.
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

#include "onnx2circle.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include <mlir/Support/FileUtilities.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Tools/ParseUtilities.h>
#include <llvm/Support/Path.h>

#include <iostream>
#include <string>

namespace onnx2circle
{

// from ONNX-MLIR src/Compiler/CompilerUtils.cpp
void registerDialects(mlir::MLIRContext &context)
{
  context.getOrLoadDialect<mlir::func::FuncDialect>();
}

int loadONNX(const std::string &onnx_path, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  llvm::StringRef inputFilename(onnx_path);
  std::string errorMessage;
  if (inputFilename.endswith(".mlir"))
  {
    // TODO implement
  }
  else if (inputFilename.endswith(".onnx"))
  {
    // TODO implement
  }
  else
  {
    llvm::errs() << "Unknown model file extension.\n";
    llvm::errs().flush();
    return -1;
  }

  return 0;
}

int convertToCircle(const O2Cparam &param)
{
  const std::string &sourcefile = param.sourcefile;

  mlir::MLIRContext context;
  registerDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto result = loadONNX(sourcefile, context, module);
  if (result != 0)
    return result;

  // TODO add processing

  return 0;
}

} // namespace onnx2circle

// NOTE sync version number with 'infra/debian/*/changelog' for upgrade
const char *__version = "0.2.0";

int entry(const O2Cparam &param)
{
  int result = onnx2circle::convertToCircle(param);
  return result;
}
