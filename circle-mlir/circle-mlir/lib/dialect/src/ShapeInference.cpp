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

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::inferShapes()
{
  AddOp op = *this;
  SmallVector<int64_t, 4> inferred;
  if (!inferBinShapes<AddOp>(op, inferred))
    return;

  auto input0_op = getOperand(0);
  auto input0_type = input0_op.getType().cast<TensorType>();
  RankedTensorType inferred_type = RankedTensorType::get(inferred, input0_type.getElementType());
  getResult().setType(inferred_type);
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

void CustomOp::inferShapes()
{
  CustomOp op = *this;
  auto outputs = op.getOutput();
  bool all_static = true;
  for (auto output : outputs)
  {
    auto output_type = output.getType().cast<ShapedType>();
    if (not output_type.hasStaticShape())
    {
      all_static = false;
      break;
    }
  }
  if (all_static)
    return;

  if (op.getCustomCode() == "Erf")
  {
    assert(op.getInput().size() == 1);
    assert(op.getOutput().size() == 1);

    auto input_op = getOperand(0);
    auto input_type = input_op.getType().cast<TensorType>();
    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t, 4> inferred(input_shape.begin(), input_shape.end());

    dumpShape<CustomOp>(op, inferred);

    RankedTensorType inferred_type = RankedTensorType::get(inferred, input_type.getElementType());
    getResult(0).setType(inferred_type);
  }
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool GetReshapeOutputType(const Value input, const Value shape, RankedTensorType &output_ty)
{
  auto input_ty = input.getType().cast<TensorType>();
  auto element_ty = input_ty.getElementType();
  auto shape_ty = shape.getType().dyn_cast<RankedTensorType>();
  if (!shape_ty)
    return false;

  if (shape_ty.getRank() != 1)
  {
    assert(false);
    return false;
  }

  mlir::DenseIntElementsAttr shape_attr;
  if (!matchPattern(shape, m_Constant(&shape_attr)))
  {
    // If only shape of `shape` is known, return ranked but dynamic output shape.
    if (shape_ty.hasStaticShape())
    {
      llvm::SmallVector<int64_t, 8> dynamic_shape(shape_ty.getDimSize(0), ShapedType::kDynamic);
      output_ty = GetTypeFromTensorShape(dynamic_shape, element_ty);
      return true;
    }
    return false;
  }

  auto input_shape = input_ty.getShape();
  llvm::SmallVector<int64_t, 4> input_sh_vals(input_shape.begin(), input_shape.end());

  // Detect if reshape output shape is folded.
  int unknown_index = -1;
  // The product of constant shape argument excluding unknown dimension.
  int64_t shape_ty_size = 1;
  llvm::SmallVector<int64_t, 8> output_ty_shape;
  output_ty_shape.reserve(shape_attr.getNumElements());
  int64_t dim_index = 0;
  for (const auto &dim : llvm::enumerate(shape_attr.getValues<APInt>()))
  {
    int64_t size = dim.value().getSExtValue();
    if (size == kDynamicSize || size == ShapedType::kDynamic)
    {
      if (unknown_index != -1)
      {
        LLVM_DEBUG({
          llvm::dbgs() << "Got two or more unknown: ";
          llvm::dbgs() << unknown_index << " and " << dim.index();
          llvm::dbgs() << "\n";
        });
        return false;
      }
      unknown_index = dim.index();
    }
    else if (size == 0)
    {
      if (dim_index < input_sh_vals.size())
        size = input_sh_vals[dim_index];
      else
      {
        // stop for debug version to find a mdoel with this case
        assert(false);
        size = 1;
      }
      shape_ty_size *= size;
    }
    else if (size > 0)
    {
      shape_ty_size *= size;
    }
    else
    {
      LLVM_DEBUG({
        llvm::dbgs() << "Got invalid size ";
        llvm::dbgs() << size << " at " << dim.index();
        llvm::dbgs() << "\n";
      });
      return false;
    }
    output_ty_shape.push_back(size);
    dim_index++;
  }

  if (!input_ty.hasStaticShape())
  {
    output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);
    return true;
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1)
  {
    // Compute number of elements in tensor shape.
    int64_t input_ty_size = 1;
    for (const auto &dim : input_ty.getShape())
    {
      // stop for debug version to find this case
      assert(dim > 0);
      if (dim > 0)
      {
        input_ty_size *= dim;
      }
    }

    const int64_t missing_dim = input_ty_size / shape_ty_size;
    if (shape_ty_size * missing_dim != input_ty_size)
    {
      LLVM_DEBUG({
        llvm::dbgs() << "requires 'input' number of elements be a multiple of ";
        llvm::dbgs() << shape_ty_size << ", but got " << input_ty_size;
        llvm::dbgs() << "\n";
      });
      return false;
    }

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }

  output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);
  return true;
}

void ReshapeOp::inferShapes()
{
  ReshapeOp op = *this;
  auto output_type = op.getOutput().getType().cast<ShapedType>();
  if (output_type.hasStaticShape())
    return;

  const Value input = op.getInput();
  const Value shape = op.getShape();

  RankedTensorType inferred_type;
  if (GetReshapeOutputType(input, shape, inferred_type))
  {
    auto output_shape = inferred_type.getShape();
    dumpShape<ReshapeOp>(op, output_shape);

    getResult().setType(inferred_type);
  }
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::inferShapes()
{
  TransposeOp op = *this;
  auto output_type = op.getOutput().getType().cast<ShapedType>();
  if (output_type.hasStaticShape())
    return;

  auto input_type = op.getInput().getType().cast<ShapedType>();
  auto perm_type = op.getPerm().getType().cast<ShapedType>();

  if (input_type.hasStaticShape() && perm_type.hasStaticShape())
  {
    if (perm_type.getNumElements() != input_type.getRank())
    {
      return;
    }
  }

  mlir::DenseIntElementsAttr perm;
  if (!matchPattern(op.getPerm(), m_Constant(&perm)))
  {
    return;
  }

  llvm::SmallVector<int64_t, 4> perm_list;
  for (const auto &perm_element : perm.getValues<APInt>())
  {
    const int64_t val = perm_element.getSExtValue();
    perm_list.push_back(val);
  }

  // Get transposed shape and set it to the output type
  if (input_type.hasStaticShape() && !output_type.hasStaticShape())
  {
    llvm::SmallVector<int64_t, 4> transposed_shape;
    for (int64_t axis : perm_list)
    {
      transposed_shape.push_back(input_type.getDimSize(axis));
    }

    dumpShape<TransposeOp>(op, transposed_shape);

    auto inferred_type =
      mlir::Circle::GetTypeFromTensorShape(transposed_shape, input_type.getElementType());
    getResult().setType(inferred_type);
  }
}

} // namespace Circle
} // namespace mlir
