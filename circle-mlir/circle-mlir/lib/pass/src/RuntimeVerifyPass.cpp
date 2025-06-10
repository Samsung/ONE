/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/transforms/runtime_verify.cc

#include "RuntimeVerifyPass.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <circle-mlir/dialect/CircleDialect.h>

namespace mlir
{
namespace Circle
{

struct RuntimeVerifyPass
  : public mlir::PassWrapper<RuntimeVerifyPass, mlir::OperationPass<mlir::func::FuncOp>>
{
  RuntimeVerifyPass() = default;
  RuntimeVerifyPass(const RuntimeVerifyPass &pass)
    : mlir::PassWrapper<RuntimeVerifyPass, OperationPass<mlir::func::FuncOp>>()
  {
    // Do nothing
  }

  llvm::StringRef getArgument() const override { return "circle-runtime-verify"; }
  llvm::StringRef getDescription() const override { return "Circle Runtime Verify"; }

  void runOnOperation(void) final;
};

void RuntimeVerifyPass::runOnOperation(void)
{
  getOperation().walk([&](CirRuntimeVerifyOpInterface op) {
    if (mlir::failed(op.VerifyCirRuntimeConstraints(op.getOperation(), true)))
      signalPassFailure();
  });
}

// Verifies circle runtime constraints.
std::unique_ptr<mlir::Pass> CreateRuntimeVerifyPass(void)
{
  return std::make_unique<RuntimeVerifyPass>();
}

} // namespace Circle
} // namespace mlir
