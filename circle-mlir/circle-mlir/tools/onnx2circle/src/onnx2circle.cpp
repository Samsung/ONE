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

// ONNX-MLIR
#include <src/Dialect/ONNX/ONNXDialect.hpp>
#include <src/Builder/FrontendDialectTransformer.hpp>
#include <src/Compiler/CompilerOptions.hpp>

// CIRCLE-MLIR
#include <circle-mlir/dialect/CircleDialect.h>
#include <circle-mlir/pass/CirclePass.h>
#include <circle-mlir/export/CircleExport.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace onnx2circle
{

// from ONNX-MLIR src/Compiler/CompilerUtils.cpp
std::string dirName(llvm::StringRef inputFilename)
{
  llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
  llvm::sys::path::remove_filename(path);
  return std::string(path.data(), path.size());
}

// from ONNX-MLIR src/Compiler/CompilerUtils.cpp
void registerDialects(mlir::MLIRContext &context)
{
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  context.getOrLoadDialect<mlir::ONNXDialect>();
  context.getOrLoadDialect<mlir::Circle::CIRDialect>();
}

int loadONNX(const std::string &onnx_path, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  llvm::StringRef inputFilename(onnx_path);
  std::string errorMessage;
  if (inputFilename.ends_with(".mlir"))
  {
    auto input = mlir::openInputFile(inputFilename, &errorMessage);
    if (!input)
    {
      llvm::errs() << errorMessage << "\n";
      llvm::errs().flush();
      return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module)
    {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      llvm::errs().flush();
      return -1;
    }
  }
  else if (inputFilename.ends_with(".onnx"))
  {
    onnx_mlir::ImportOptions options;
    options.useOnnxModelTypes = onnx_mlir::useOnnxModelTypes;
    options.invokeOnnxVersionConverter = onnx_mlir::invokeOnnxVersionConverter;
    options.shapeInformation = onnx_mlir::shapeInformation;
    options.allowSorting = onnx_mlir::allowSorting;
    options.externalDataDir = dirName(inputFilename);

    int rc =
      onnx_mlir::ImportFrontendModelFile(inputFilename, context, module, &errorMessage, options);
    if (rc != onnx_mlir::CompilerSuccess)
    {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      llvm::errs() << errorMessage << "\n";
      llvm::errs().flush();
      return -1;
    }
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
  const std::string &targetfile = param.targetfile;

  mlir::MLIRContext context;
  registerDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto result = loadONNX(sourcefile, context, module);
  if (result != 0)
    return result;

  if (param.check_rawprocessed)
  {
    std::string serialized_flatbuffer;
    if (mlir::Circle::MlirToFlatBufferTranslateFunction(module.get(), &serialized_flatbuffer))
    {
      std::string error_msg;
      std::filesystem::path tempFile = targetfile;
      tempFile.replace_extension(".raw.circle");
      auto output = mlir::openOutputFile(tempFile.string(), &error_msg);
      // TODO error handle
      output->os() << serialized_flatbuffer;
      output->keep();
    }
  }

  if (param.dynamic_batch_to_single_batch)
  {
    result = mlir::Circle::dynamicBatchToSingleBatch(context, module);
    if (result != 0)
      return result;
  }

  result = mlir::Circle::preprocessONNX(context, module);
  if (result != 0)
    return result;

  result = mlir::Circle::shapeInferenceONNX(context, module);
  if (result != 0)
    return result;

  result = mlir::Circle::convertToCircle(context, module);
  if (result != 0)
    return result;

  result = mlir::Circle::postProcessCircle(context, module);
  if (result != 0)
    return result;

  if (param.check_shapeinf)
  {
    result = mlir::Circle::shapeValidateCircle(context, module);
    if (result != 0)
      return result;
  }
  if (param.check_dynshapeinf)
  {
    // output should have any static shape from dynamic input
    result = mlir::Circle::dynaShapeValidateCircle(context, module);
    if (result != 0)
      return result;
  }

  std::string error_msg;
  if (param.save_ops)
  {
    std::string output_filename = targetfile + ".ops";
    auto output = mlir::openOutputFile(output_filename, &error_msg);
    if (!error_msg.empty())
    {
      llvm::errs() << "Failed: " << error_msg << "\n";
      return -1;
    }
    result = mlir::Circle::dumpCircleOps(output->os(), context, module);
    if (result == 0)
      output->keep();

    return result;
  }

  std::string serialized_flatbuffer;
  if (!mlir::Circle::MlirToFlatBufferTranslateFunction(module.get(), &serialized_flatbuffer))
    return -1;
  auto output = mlir::openOutputFile(targetfile, &error_msg);
  // TODO error handle
  output->os() << serialized_flatbuffer;
  output->keep();

  return 0;
}

} // namespace onnx2circle

// NOTE sync version number with 'infra/debian/*/changelog' for upgrade
const char *__version = "0.4.0";

int entry(const O2Cparam &param)
{
  int result = onnx2circle::convertToCircle(param);
  return result;
}
