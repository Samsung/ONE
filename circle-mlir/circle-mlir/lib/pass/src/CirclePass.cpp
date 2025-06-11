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

#include "circle-mlir/pass/CirclePass.h"

#include "DynamicBatchToSingleBatchPass.h"
#include "ConvertONNXToCirclePass.h"
#include "RewriteCirclePass.h"
#include "RewriteONNXPass.h"
#include "DumpCircleOpsPass.h"
#include "RuntimeVerifyPass.h"
#include "ShapeInferencePass.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <src/Pass/Passes.hpp>

namespace mlir
{
namespace Circle
{

namespace
{

// TODO move to somewhere common
template <typename T> T safecast(const char *, const T &);

template <> int safecast<int>(const char *s, const int &value)
{
  return (s == nullptr) ? value : std::stoi(s);
}

} // namespace

int preprocessONNX(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int dump = safecast<int>(std::getenv("CM_ONNX_DUMP"), 0);
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintBeforePass;
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintAfterPass;
  shouldPrintBeforePass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  shouldPrintAfterPass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, false, false, false,
                      llvm::errs());

  int result = 0;
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  // Replace ONNXReturnOp with func::ReturnOp.
  pm.addPass(onnx_mlir::createStandardFuncReturnPass());
  pm.addPass(createRewriteONNXPass());
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
  {
    // TODO show error message if needed
    result = -1;
  }

  return result;
}

int shapeInferenceONNX(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int result = 0;
  pm.addPass(onnx_mlir::createShapeInferencePass());
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
  {
    // TODO show error message if needed
    result = -1;
  }

  return result;
}

int convertToCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int dump = safecast<int>(std::getenv("CM_PASS_DUMP"), 0);
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintBeforePass;
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintAfterPass;
  shouldPrintBeforePass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  shouldPrintAfterPass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, false, false, false,
                      llvm::errs());

  int result = 0;
  pm.addPass(createConvertONNXToCirclePass());
  pm.addPass(CreateRuntimeVerifyPass());
  pm.addPass(mlir::createCanonicalizerPass());
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
    result = -1;

  return result;
}

int postProcessCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int dump = safecast<int>(std::getenv("CM_PASS_DUMP"), 0);
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintBeforePass;
  std::function<bool(mlir::Pass *, mlir::Operation *)> shouldPrintAfterPass;
  shouldPrintBeforePass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  shouldPrintAfterPass = [&](mlir::Pass *, mlir::Operation *) { return dump ? true : false; };
  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, false, false, false,
                      llvm::errs());

  int result = 0;
  int64_t dyna_count = 0;
  int64_t prev_count = 0;
  int64_t same_count = 0;
  pm.addPass(CreateShapeInferencePass(dyna_count));
  pm.addPass(mlir::createCanonicalizerPass()); // to make Op::fold() call
  pm.addPass(createRewriteCirclePass());
  pm.addPass(CreateRuntimeVerifyPass());
  // NOTE as couldn't find how to make pm.run() continue till there is nothing to shape infer,
  //      this loop is to emulate that kind of feature.
  //      1/ loop until all has become static shape
  //      2/ continue loop while dynamic shape node count has changed
  //      3/ exit loop if count has not changed for 20 times, to prevent infinite loops
  // TODO find a correct way to continue shape infer.
  while (same_count < 20)
  {
    dyna_count = 0;
    auto runres = pm.run(*module);
    if (mlir::failed(runres))
    {
      result = -1;
      break;
    }
    if (dyna_count == 0)
      break;
    if (dyna_count == prev_count)
      same_count++;
    else if (dyna_count < prev_count)
      same_count = 0;

    prev_count = dyna_count;
  }

  return result;
}

int shapeValidateCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int result = 0;
  pm.addPass(CreateShapeValidatePass());
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
    result = -1;

  return result;
}

int dynaShapeValidateCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  int result = 0;
  pm.addPass(CreateDynaShapeValidatePass());
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
    result = -1;

  return result;
}

int dumpCircleOps(llvm::raw_fd_ostream &os, mlir::MLIRContext &context,
                  mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  DumpCircleOpsPass::GetOStream_t gos = [&](void) -> llvm::raw_fd_ostream & { return os; };

  int result = 0;
  auto pass = std::make_unique<mlir::Circle::DumpCircleOpsPass>();
  pass->ostream(gos);
  pm.addPass(std::move(pass));
  auto runres = pm.run(*module);
  if (mlir::failed(runres))
    result = -1;

  return result;
}

} // namespace Circle
} // namespace mlir
