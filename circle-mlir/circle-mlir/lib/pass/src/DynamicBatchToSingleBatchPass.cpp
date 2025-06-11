/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019-2024 The IBM Research Authors.
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

// NOTE logic of this code is from onnx-mlir/src/Compiler/CompilerUtils.cpp

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

namespace mlir
{
namespace Circle
{

struct DynamicBatchToSingleBatchPass
  : public mlir::PassWrapper<DynamicBatchToSingleBatchPass, mlir::OperationPass<mlir::func::FuncOp>>
{
  DynamicBatchToSingleBatchPass() = default;
  DynamicBatchToSingleBatchPass(const DynamicBatchToSingleBatchPass &pass)
    : mlir::PassWrapper<DynamicBatchToSingleBatchPass, OperationPass<mlir::func::FuncOp>>()
  {
    // Do nothing
  }

  llvm::StringRef getArgument() const override { return "dynamic-batch-to-single"; }

  llvm::StringRef getDescription() const override
  {
    return "Convert dynamic batch size (first dimension) of inputs to 1";
  }

  Option<std::string> target{
    *this, "target",
    ::llvm::cl::desc("Convert dynamic batch size (first dimension) of inputs to 1"),
    ::llvm::cl::init("")};

  void runOnOperation() final;
};

void DynamicBatchToSingleBatchPass::runOnOperation()
{
  mlir::func::FuncOp funcOp = getOperation();
  mlir::MLIRContext *context = &getContext();

  // dynamic shape of func to static shape
  auto funcType = mlir::dyn_cast<mlir::FunctionType>(funcOp.getFunctionType());
  mlir::ArrayRef<mlir::Type> argTypes = funcType.getInputs();
  mlir::SmallVector<Type, 4> newArgTypes;
  for (uint64_t i = 0; i < argTypes.size(); ++i)
  {
    mlir::Type argTy = argTypes[i];

    if (auto rankedTensorTy = mlir::dyn_cast<mlir::RankedTensorType>(argTy))
    {
      // NOTE although we only modify origDims[0], this loop is prepared
      // for potential extension to modify all dynamic dimensions.
      mlir::ArrayRef<int64_t> origDims = rankedTensorTy.getShape();
      mlir::SmallVector<int64_t, 4> staticDims;
      for (uint64_t i = 0; i < origDims.size(); ++i)
      {
        LLVM_DEBUG({
          llvm::dbgs() << "Input " << i << ": ";
          llvm::dbgs() << origDims[i];
          llvm::dbgs() << "\n";
        });
        // Only update to 1 if first(batch) dim is dynamic
        bool first_dynamic = (i == 0 && mlir::ShapedType::isDynamic(origDims[i]));
        staticDims.emplace_back(first_dynamic ? 1 : origDims[i]);
      }
      argTy = mlir::RankedTensorType::get(staticDims, rankedTensorTy.getElementType());
    }
    // update the argument
    funcOp.getBody().back().getArgument(i).setType(argTy);
    newArgTypes.emplace_back(argTy);
  }
  // Update the function type
  auto newType = mlir::FunctionType::get(context, newArgTypes, funcType.getResults());
  funcOp.setType(newType);
}

std::unique_ptr<mlir::Pass> createDynamicBatchToSingleBatchPass(void)
{
  return std::make_unique<mlir::Circle::DynamicBatchToSingleBatchPass>();
}

} // namespace Circle
} // namespace mlir
