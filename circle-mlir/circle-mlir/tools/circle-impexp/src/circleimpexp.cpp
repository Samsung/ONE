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

#include "circleimpexp.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

#include <mlir/Support/FileUtilities.h>
#include <llvm/Support/ToolOutputFile.h>

#include <circle-mlir/dialect/CircleDialect.h>
#include <circle-mlir/import/CircleImport.h>
#include <circle-mlir/export/CircleExport.h>

#include <fstream>
#include <vector>

class FileLoader
{
private:
  using DataBuffer = std::vector<char>;

public:
  explicit FileLoader(const std::string &path) : _path(path) {}

public:
  FileLoader(const FileLoader &) = delete;
  FileLoader &operator=(const FileLoader &) = delete;

public:
  DataBuffer load(void) const
  {
    std::ifstream file(_path, std::ios::binary | std::ios::in);
    if (!file.good())
    {
      std::string errmsg = "Failed to open file: " + _path;
      throw std::runtime_error(errmsg.c_str());
    }

    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    DataBuffer data(fileSize);

    // read the data
    file.read(data.data(), fileSize);
    if (file.fail())
    {
      std::string errmsg = "Failed to read file: " + _path;
      throw std::runtime_error(errmsg.c_str());
    }

    return data;
  }

private:
  const std::string _path;
};

int circleImport(const std::string &sourcefile, std::string &model_string)
{
  FileLoader file_loader{sourcefile};
  std::vector<char> model_data;

  try
  {
    model_data = file_loader.load();
  }
  catch (const std::runtime_error &err)
  {
    llvm::errs() << err.what() << "\n";
    llvm::errs().flush();
    return -1;
  }

  flatbuffers::Verifier verifier{reinterpret_cast<uint8_t *>(model_data.data()), model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    llvm::errs() << "Invalid input file '" << sourcefile << "'\n";
    llvm::errs().flush();
    return -1;
  }

  model_string.insert(0, reinterpret_cast<const char *>(model_data.data()), model_data.size());
  return 0;
}

void registerDialects(mlir::MLIRContext &context)
{
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::Circle::CIRDialect>();
}

int circleImportExport(const CirImpExpParam &param)
{
  const std::string &sourcefile = param.sourcefile;
  const std::string &targetfile = param.targetfile;

  std::string model_string;
  int result = circleImport(sourcefile, model_string);
  if (result < 0)
    return result;

  mlir::MLIRContext context;
  registerDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
    mlir::Circle::FlatBufferToMlir(model_string, &context, mlir::UnknownLoc::get(&context));
  if (!module)
  {
    llvm::errs() << "Error can't load file " << sourcefile << "\n";
    llvm::errs().flush();
    return -1;
  }

  std::string serialized_flatbuffer;
  if (!mlir::Circle::MlirToFlatBufferTranslateFunction(module.get(), &serialized_flatbuffer))
    return -1;
  std::string error_msg;
  auto output = mlir::openOutputFile(targetfile, &error_msg);
  // TODO error handle
  output->os() << serialized_flatbuffer;
  output->keep();

  return 0;
}

int entry(const CirImpExpParam &param)
{
  int result = circleImportExport(param);
  return result;
}
