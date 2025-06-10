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

#include "RewriteCirclePass.h"
#include "ConvertHelper.h"

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// Optimizations
#include "opt/ConvertDivToMul.h"
#include "opt/ConvertMirrorPadPad32.h"
#include "opt/ConvertReshapeShape32.h"
#include "opt/ConvertResizeBilinearSize32.h"
#include "opt/ConvertResizeNearestSize32.h"
#include "opt/ConvertSqrtDivToRsqrt.h"
#include "opt/FuseAddRelu.h"
#include "opt/FuseConv2DRelu.h"
#include "opt/FuseFullyConnectedAdd.h"

namespace mlir
{
namespace Circle
{

struct RewriteCirclePass
  : public mlir::PassWrapper<RewriteCirclePass, mlir::OperationPass<mlir::func::FuncOp>>
{
  RewriteCirclePass() = default;
  RewriteCirclePass(const RewriteCirclePass &pass)
    : mlir::PassWrapper<RewriteCirclePass, OperationPass<mlir::func::FuncOp>>()
  {
    // Do nothing
  }

  llvm::StringRef getArgument() const override { return "circle-rewrite"; }

  llvm::StringRef getDescription() const override { return "Rewrite Circle ops"; }

  Option<std::string> target{*this, "target",
                             ::llvm::cl::desc("Rewrite Circle dialect to Circle dialect"),
                             ::llvm::cl::init("")};

  void runOnOperation() final;

private:
  // Apply canonicalization, mainly constant folding, on the function.
  void applyCanonicalization();
  // Apply activation fusion
  void applyActivationFusion();
};

void RewriteCirclePass::runOnOperation()
{
  // canonicalization
  applyCanonicalization();
  // activation fusion
  applyActivationFusion();
}

void RewriteCirclePass::applyCanonicalization()
{
  mlir::func::FuncOp func = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::RewritePatternSet patterns(context);

  func->walk([&](Operation *op) {
    op->getRegisteredInfo()->getCanonicalizationPatterns(patterns, context);
  });
}

void RewriteCirclePass::applyActivationFusion()
{
  mlir::func::FuncOp func = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::RewritePatternSet patterns(context);

  // TODO enable Tanh after circle-interpreter works
  // patterns.add<FuseConv2DRelu<TanhOp, ACT_TANH>>(context);

  patterns.add<FuseAddRelu<ReluOp, ACT_RELU>>(context);
  patterns.add<FuseAddRelu<Relu6Op, ACT_RELU6>>(context);
  patterns.add<FuseConv2DRelu<ReluOp, ACT_RELU>>(context);
  patterns.add<FuseConv2DRelu<Relu6Op, ACT_RELU6>>(context);
  patterns.add<FuseFullyConnectedAdd>(context);

  patterns.add<ConvertDivToMul>(context);
  patterns.add<ConvertMirrorPadPad32>(context);
  patterns.add<ConvertReshapeShape32>(context);
  patterns.add<ConvertResizeBilinearSize32>(context);
  patterns.add<ConvertResizeNearestSize32>(context);
  patterns.add<ConvertSqrtDivToRsqrt>(context);

  // TODO add more patterns

  (void)applyPatternsGreedily(func, std::move(patterns));
}

std::unique_ptr<mlir::Pass> createRewriteCirclePass(void)
{
  return std::make_unique<mlir::Circle::RewriteCirclePass>();
}

} // namespace Circle
} // namespace mlir
