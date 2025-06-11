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

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include "ShapeInferencePass.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <circle-mlir/dialect/CircleDialect.h>

// from tensorflow/core/transforms/shape_inference/pass.cc

namespace mlir
{
namespace Circle
{

/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
struct ShapeInferencePass
  : public mlir::PassWrapper<ShapeInferencePass, mlir::OperationPass<mlir::func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  void runOnOperation() override
  {
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    int64_t op_count = 0;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
      {
        opWorklist.insert(op);
        op_count++;
      }
    });

    // TODO remove this when this pass runs again if there is any change in the graph
    if (_dynacount)
      *_dynacount = op_count;

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty())
    {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<CirShapeInference>(op))
      {
        shapeOp.inferShapes();
        if (returnsDynamicShape(op))
        {
          LLVM_DEBUG({
            mlir::Location opLoc = op->getLoc();
            llvm::dbgs() << "-- " << opLoc << " still has dynamic shape\n";
          });
        }
      }
      else
      {
        LLVM_DEBUG({
          mlir::Location opLoc = op->getLoc();
          llvm::dbgs() << "-- " << opLoc << " has dynamic shape but no CirShapeInference\n";
        });
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty())
    {
      f.emitWarning("Shape inference still has dynamic shapes, ")
        << opWorklist.size() << " operations couldn't be inferred\n";
      while (!opWorklist.empty())
      {
        Operation *op = *opWorklist.begin();
        LLVM_DEBUG(llvm::dbgs() << "Shape inference left: " << *op << "\n");
        opWorklist.erase(op);
      }
    }

    // set function shape to that from last op.
    // this is to update when function shape is unknown at beginning and then
    // fixed to known with shape inference.
    Operation *returnOp = f.getBody().back().getTerminator();
    assert(returnOp && "function must return");
    FunctionType fty = f.getFunctionType();
    assert(f.getNumResults() == returnOp->getNumOperands() &&
           "returned results count much match function type");
    f.setType(fty.clone(fty.getInputs(), returnOp->getOperandTypes()));
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation *op)
  {
    return llvm::all_of(op->getOperands(), [](mlir::Value operand) {
      // ignore for NoValueOp
      auto no_value = dyn_cast_or_null<mlir::Circle::NoValueOp>(operand.getDefiningOp());
      if (no_value)
        return true;
      auto resType = operand.getType();
      return llvm::isa<RankedTensorType>(resType);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation *op)
  {
    // return false to ignore for NoValueOp, as it doesn't have ShapedType
    auto no_value = dyn_cast_or_null<mlir::Circle::NoValueOp>(*op);
    if (no_value)
      return false;

    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      // Checks whether each dimension is all dynamic if it is ShapedType.
      if (ShapedType shapedType = mlir::dyn_cast<ShapedType>(resultType))
      {
        if (not shapedType.hasRank())
          return true;
        int rank = shapedType.getRank();
        for (int i = 0; i < rank; ++i)
          if (shapedType.isDynamicDim(i))
            return true;
        return false;
      }
      // Non-shaped types are considered dynamic
      return true;
    });
  }

  int64_t *_dynacount = nullptr;
};

struct ShapeValidatePass
  : public mlir::PassWrapper<ShapeValidatePass, mlir::OperationPass<mlir::func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeValidatePass)
  void runOnOperation() override
  {
    auto f = getOperation();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    if (!opWorklist.empty())
    {
      f.emitError("Shape validation found node with unknown shape.\n");
      // TODO dump ops when necessary
      signalPassFailure();
    }
  }

  static bool returnsDynamicShape(Operation *op)
  {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      if (ShapedType shapedType = mlir::dyn_cast<ShapedType>(resultType))
      {
        int rank = shapedType.getRank();
        for (int i = 0; i < rank; ++i)
          if (shapedType.isDynamicDim(i))
            return true;
      }
      return false;
    });
  }
};

struct AnyShapeValidatePass
  : public mlir::PassWrapper<AnyShapeValidatePass, mlir::OperationPass<mlir::func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnyShapeValidatePass)
  void runOnOperation() override
  {
    auto f = getOperation();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsFullDynamicShape(op))
        opWorklist.insert(op);
    });

    if (!opWorklist.empty())
    {
      f.emitError("Shape validation found node with full dynamic shape.\n");
      // TODO dump ops when necessary
      signalPassFailure();
    }
  }

  static bool returnsFullDynamicShape(Operation *op)
  {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      if (ShapedType shapedType = mlir::dyn_cast<ShapedType>(resultType))
      {
        int rank = shapedType.getRank();
        for (int i = 0; i < rank; ++i)
          if (not shapedType.isDynamicDim(i))
            return false;
        return true;
      }
      return false;
    });
  }
};

// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> CreateShapeInferencePass(int64_t &dynaCount)
{
  auto inst = std::make_unique<ShapeInferencePass>();
  inst->_dynacount = &dynaCount;
  return inst;
}

std::unique_ptr<mlir::Pass> CreateShapeValidatePass(void)
{
  return std::make_unique<ShapeValidatePass>();
}

// test helper to check input model having output with all dynamic dim
// if output has any static dim, it is success
std::unique_ptr<mlir::Pass> CreateDynaShapeValidatePass(void)
{
  return std::make_unique<AnyShapeValidatePass>();
}

} // namespace Circle
} // namespace mlir
