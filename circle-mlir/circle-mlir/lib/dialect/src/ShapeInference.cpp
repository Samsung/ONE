/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "circle-mlir/dialect/CircleDialect.h"

#include "utils/DynamicShapeUtils.h"
#include "utils/Padding.h"

#include <mlir/IR/Matchers.h>

namespace mlir
{
namespace Circle
{

// To reuse calculation for shape inference from CircleDialect.cpp
// TODO relocate to some header
LogicalResult ComputeConvWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                            int64_t dilation_rate, int64_t stride,
                                            Circle::Padding padding, int64_t *output_size);

namespace
{

bool extractElements(ConstOp &const_op, std::vector<int64_t> &values)
{
  mlir::DenseElementsAttr dataAttr =
    const_op.getValueAttr().dyn_cast_or_null<mlir::DenseElementsAttr>();
  if (dataAttr == nullptr)
    return false;
  if (!dataAttr.getElementType().isa<mlir::IntegerType>())
    return false;

  for (auto value : dataAttr.getValues<llvm::APInt>())
    values.push_back(value.getSExtValue());
  return true;
}

template <typename OP> void dumpShape(OP op, const llvm::ArrayRef<int64_t> &inferred)
{
  LLVM_DEBUG({
    mlir::Location opLoc = op->getLoc();
    llvm::dbgs() << "-- " << typeid(OP).name() << " " << opLoc << " shape-inf: ";
    for (size_t i = 0; i < inferred.size(); ++i)
    {
      llvm::dbgs() << inferred[i];
      if (i < inferred.size() - 1)
        llvm::dbgs() << ",";
    }
    llvm::dbgs() << "\n";
  });
}

} // namespace

namespace
{

template <typename BINOP> bool inferBinShapes(BINOP &op, SmallVector<int64_t, 4> &inferred)
{
  auto out_type = op.getOutput().getType().template cast<ShapedType>();
  if (out_type.hasStaticShape())
    return false;

  auto inp0_op = op.getOperand(0);
  auto inp0_type = inp0_op.getType().template cast<TensorType>();
  auto inp1_op = op.getOperand(1);
  auto inp1_type = inp1_op.getType().template cast<TensorType>();

  if (!OpTrait::util::getBroadcastedShape(inp0_type.getShape(), inp1_type.getShape(), inferred))
    return false;

  dumpShape<BINOP>(op, inferred);

  return true;
}

} // namespace

// TODO add AddOp

} // namespace Circle
} // namespace mlir
