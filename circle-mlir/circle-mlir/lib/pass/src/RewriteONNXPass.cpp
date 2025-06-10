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

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include "RewriteONNXPass.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// from onnx-mlir source
#include <src/Dialect/ONNX/ONNXOps.hpp>

#include "onnx/CompactReshapeConvReshape.h"

namespace mlir
{
namespace Circle
{

struct RewriteONNXPass
  : public mlir::PassWrapper<RewriteONNXPass, mlir::OperationPass<mlir::func::FuncOp>>
{
  RewriteONNXPass() = default;
  RewriteONNXPass(const RewriteONNXPass &pass)
    : mlir::PassWrapper<RewriteONNXPass, OperationPass<mlir::func::FuncOp>>()
  {
    // Do nothing
  }

  llvm::StringRef getArgument() const override { return "circle-onnx-rewrite"; }

  llvm::StringRef getDescription() const override { return "Rewrite ONNX ops"; }

  Option<std::string> target{*this, "target",
                             ::llvm::cl::desc("Rewrite ONNX dialect to ONNX dialect"),
                             ::llvm::cl::init("")};

  void runOnOperation() final;
};

void RewriteONNXPass::runOnOperation()
{
  mlir::func::FuncOp func = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::RewritePatternSet patterns(context);

  patterns.add<CompactReshapeConvReshape>(context); // remove unnecessary reshapes
  // TODO add more patterns

  (void)applyPatternsGreedily(func, std::move(patterns));
}

std::unique_ptr<mlir::Pass> createRewriteONNXPass(void)
{
  return std::make_unique<mlir::Circle::RewriteONNXPass>();
}

} // namespace Circle
} // namespace mlir
