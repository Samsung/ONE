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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <cassert>
#include <numeric>

#include "circle-mlir/dialect/CircleDialect.h"
#include "circle-mlir/dialect/NameUtils.h"
#include "utils/DynamicShapeUtils.h"
#include "utils/KernelShapeUtil.h"
#include "utils/Padding.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Threading.h>

#include <mlir/Dialect/Arith/IR/Arith.h>   // from @llvm-project
#include <mlir/Dialect/Func/IR/FuncOps.h>  // from @llvm-project
#include <mlir/IR/Attributes.h>            // from @llvm-project
#include <mlir/IR/Matchers.h>              // from @llvm-project
#include <mlir/IR/OpImplementation.h>      // from @llvm-project
#include <mlir/IR/PatternMatch.h>          // from @llvm-project
#include <mlir/IR/TypeUtilities.h>         // from @llvm-project
#include <mlir/IR/Types.h>                 // from @llvm-project
#include <mlir/Support/LogicalResult.h>    // from @llvm-project
#include <mlir/Transforms/FoldUtils.h>     // from @llvm-project
#include <mlir/Transforms/InliningUtils.h> // from @llvm-project

#include <absl/strings/escaping.h>

namespace mlir
{
namespace Circle
{

// NOTE This generated header should be included in this namespace to prevent multiple definition
// error with 'populateWithGenerated(mlir::RewritePatternSet&)' in the library.
#include "mlir/CircleRewrite.cc.inc"

namespace
{

ParseResult parseOneResultSameOperandTypeOp(OpAsmParser &parser, OperationState &result)
{
  SmallVector<OpAsmParser::UnresolvedOperand, 2> ops;
  Type type;
  // If the operand list is in-between parentheses, then we have a generic form.
  // (see the fallback in `printOneResultOp`).
  SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalLParen())
  {
    if (parser.parseOperandList(ops) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
        parser.parseType(type))
      return failure();
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType)
    {
      parser.emitError(loc, "expected function type");
      return failure();
    }
    if (parser.resolveOperands(ops, fnType.getInputs(), loc, result.operands))
      return failure();
    result.addTypes(fnType.getResults());
    return success();
  }
  return failure(parser.parseOperandList(ops) || parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(ops, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

void printOneResultOp(Operation *op, OpAsmPrinter &p)
{
  assert(op->getNumResults() == 1 && "op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(), [&](Type type) { return type != resultType; }))
  {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  // Now we can output only one type for all operands and the result.
  p << " : " << resultType;
}

} // namespace

// Returns true when the given operand arguments have the same shape or
// broadcastable shape within the given rank. If any given shapes are
// non-static and maximum rank is within the given rank, this method returns
// true.
bool VerifyOperandsHaveSameShapesOrBroadcastableShape(Operation *op, ArrayRef<unsigned> indices,
                                                      int max_bcast_rank)
{
  if (indices.empty())
    return true;

  // First, it checks there are any inputs that has unknown rank.
  bool has_unknown_shape_input = false;
  bool has_same_shape = true;
  bool reach_first_known_shape = false;
  int64_t max_rank = -1;

  ArrayRef<int64_t> pivot_shape;
  SmallVector<int64_t, 4> current_shape;
  SmallVector<int64_t, 4> result_shape;

  for (unsigned index : indices)
  {
    ShapedType shaped_type = op->getOperand(index).getType().dyn_cast<ShapedType>();
    if (!shaped_type || !shaped_type.hasRank())
    {
      // Marks that we have an unknown rank input.
      has_unknown_shape_input = true;
      continue;
    }
    max_rank = std::max(max_rank, shaped_type.getRank());
    if (!shaped_type.hasStaticShape())
    {
      // Marks that we have an unknown shape input.
      has_unknown_shape_input = true;
      continue;
    }

    ArrayRef<int64_t> shape = shaped_type.getShape();
    if (!reach_first_known_shape)
    {
      pivot_shape = shape;
      current_shape.assign(shape.begin(), shape.end());
      reach_first_known_shape = true;
      continue;
    }

    if (!pivot_shape.equals(shape))
    {
      has_same_shape = false;
    }
    //  Checks if all the inputs are broadcastable since they have not all the
    //  same shapes.
    if (!OpTrait::util::getBroadcastedShape(current_shape, shape, result_shape))
    {
      return false;
    }
    current_shape = result_shape;
  }

  // If all the shape is known and same, CPU kernels are able to handle inputs
  // regardless of dimension size.
  if (!has_unknown_shape_input)
  {
    return has_same_shape || max_rank <= max_bcast_rank;
  }

  // It will treat the unknown shape inputs as acceptable inputs for model
  // compatibility if all known ranks are no bigger than the allowed broadcast
  // maximum rank.
  if (max_rank <= max_bcast_rank)
  {
    return true;
  }

  // TODO support broadcast

  return true;
}

// Return true when the given element_type is I32.
bool IsI32Type(Type element_type)
{
  return element_type.isInteger(32) && !element_type.isUnsignedInteger();
}

// Return true when the given element_type is I64.
bool IsI64Type(Type element_type)
{
  return element_type.isInteger(64) && !element_type.isUnsignedInteger();
}

// Return true if the value is a splat tensor constant zero.
bool EqualsZero(Value value)
{
  DenseElementsAttr constant;
  if (!matchPattern(value, m_Constant(&constant)) || !constant.isSplat())
  {
    return false;
  }

  Type element_type = value.getType().cast<ShapedType>().getElementType();
  if (element_type.isa<FloatType>())
  {
    return constant.getSplatValue<APFloat>().isZero();
  }
  else
  {
    return false;
  }
}

// Replaces the bias operand with a "none" type value if the bias value is
// constant zero.
// `ConcreteOpType` must be an concrete MLIR op class that has an optional
// bias operand named 'bias'.
template <typename ConcreteOpType>
struct RemoveOptionalZeroBias : public OpRewritePattern<ConcreteOpType>
{
  using OpRewritePattern<ConcreteOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcreteOpType op, PatternRewriter &rewriter) const override
  {
    if (EqualsZero(op.getBias()))
    {
      auto none_value = rewriter.create<mlir::Circle::NoValueOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
      op.getBiasMutable().assign(none_value);
      return success();
    }

    return failure();
  }
};

namespace
{

std::string GetOperationName(mlir::Operation *op)
{
  assert(op != nullptr);

  mlir::Location opLoc = op->getLoc();
  auto name = mlir::GetNameFromLoc(opLoc);
  if (!name.empty())
    return name;

  // Use operator type as name if there is no name
  // TODO revise this to better implementation
  static uint64_t sequence = 1;
  auto seqstr = "C" + std::to_string(sequence);
  sequence++;
  return op->getName().getStringRef().str() + seqstr;
}

DenseIntElementsAttr GetI1ElementsAttr(ArrayRef<bool> values, Builder *builder)
{
  mlir::RankedTensorType ty =
    GetTypeFromTensorShape({static_cast<int32_t>(values.size())}, builder->getI1Type(), {});
  return DenseIntElementsAttr::get(ty, values);
}

DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values, Builder *builder)
{
  mlir::RankedTensorType ty =
    GetTypeFromTensorShape({static_cast<int32_t>(values.size())}, builder->getI32Type(), {});
  return DenseIntElementsAttr::get(ty, values);
}

template <typename T> bool ExtractConstantIntValues(mlir::Value &input, std::vector<T> &values)
{
  mlir::DenseElementsAttr dataAttr;

  if (auto constOp2 = dyn_cast_or_null<mlir::Circle::ConstOp>(input.getDefiningOp()))
    dataAttr = constOp2.getValueAttr().dyn_cast_or_null<mlir::DenseElementsAttr>();
  else
    return false;

  if (dataAttr == nullptr)
    return false;

  auto valueIt = dataAttr.getValues<llvm::APInt>().begin();
  auto valueEd = dataAttr.getValues<llvm::APInt>().end();
  for (; valueIt != valueEd; ++valueIt)
  {
    T value = static_cast<T>((*valueIt).getSExtValue());
    values.push_back(value);
  }
  return true;
}

bool ExtractConstantValues(mlir::Value &input, std::vector<int64_t> &values)
{
  return ExtractConstantIntValues<int64_t>(input, values);
}

} // namespace

//===----------------------------------------------------------------------===//
// CircleDialect
//===----------------------------------------------------------------------===//

void CIRDialect::printType(Type type, DialectAsmPrinter &os) const
{
  if (type.isa<ControlType>())
  {
    os << "control";
    return;
  }
  os << "<unknown CIR type>";
}

Type CIRDialect::parseType(DialectAsmParser &parser) const
{
  StringRef data_type;
  if (parser.parseKeyword(&data_type))
    return Type();
  if (data_type == "control")
    return ControlType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown CIR type: " << data_type;
  return nullptr;
}

void CIRDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "mlir/CircleOps.cc.inc"
    >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/CircleOpsAttrdefs.cc.inc"
    >();
  addTypes<ControlType>();
}

//===----------------------------------------------------------------------===//
// Common support logic
//===----------------------------------------------------------------------===//

namespace
{

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t)
{
  if (auto shaped_type = t.dyn_cast_or_null<ShapedType>())
  {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Returns true if it is a shaped type of i64 elements.
inline bool IsI64ShapedType(Type t)
{
  if (auto shaped_type = t.dyn_cast_or_null<ShapedType>())
  {
    return shaped_type.getElementType().isInteger(64);
  }
  return false;
}

} // namespace

} // namespace Circle
} // namespace mlir

#include "ConstFold.inc.cpp"

namespace mlir
{
namespace Circle
{

namespace
{

bool InputOutputHasSameShape(mlir::Type input_type, mlir::Type output_type)
{
  auto input_shaped_type = input_type.dyn_cast_or_null<ShapedType>();
  if (!input_shaped_type || !input_shaped_type.hasStaticShape())
    return false;

  auto output_shaped_type = output_type.dyn_cast_or_null<ShapedType>();
  if (!output_shaped_type || !output_shaped_type.hasStaticShape())
    return false;

  return input_shaped_type == output_shaped_type;
}

} // namespace

//===----------------------------------------------------------------------===//
// ConstBytesAttr
//===----------------------------------------------------------------------===//

Attribute ConstBytesAttr::parse(AsmParser &parser, Type type)
{
  if (parser.parseColon())
  {
    return nullptr;
  }

  std::string data;
  if (parser.parseString(&data))
  {
    return nullptr;
  }
  if (data.size() < 2 || data.substr(0, 2) != "0x")
  {
    parser.emitError(parser.getNameLoc(), "Hex string doesn't start with `0x`");
    return nullptr;
  }

  std::string bytes_data = absl::HexStringToBytes(data.substr(2));
  return ConstBytesAttr::get(parser.getBuilder().getContext(), bytes_data);
}

void ConstBytesAttr::print(mlir::AsmPrinter &printer) const
{
  StringRef bytes_str = getValue();
  printer << " : \"0x" << llvm::toHex(bytes_str) << "\"";
}

#include "mlir/CircleOpInterface.cc.inc"
#include "mlir/CircleShapeInferenceOpInterfaces.cc.inc"

} // namespace Circle
} // namespace mlir

// TODO add AddOp
#include "ops/CustomOp.h"

#include "mlir/CircleOpsDialect.cc.inc"
#include "mlir/CircleOpsEnums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/CircleOpsAttrdefs.cc.inc"
#define GET_OP_CLASSES
#include "mlir/CircleOps.cc.inc"

namespace mlir
{
namespace Circle
{

#include "mlir/RuntimeVerifiers.inc"

Operation *CIRDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                           Location loc)
{
  // If this is a constant bytes attribute or the result type doesn't match the
  // attribute type, then generate a tfl.pseudo_const.
  if (value.isa<ConstBytesAttr>() ||
      (value.isa<ElementsAttr>() && value.cast<ElementsAttr>().getType() != type))
    return builder.create<ConstOp>(loc, type, value.cast<ElementsAttr>());
  if (ConstOp::isBuildableWith(value, type))
    return builder.create<ConstOp>(loc, type, value.cast<ElementsAttr>());
  if (NoValueOp::isBuildableWith(value, type))
    return builder.create<NoValueOp>(loc, type, value.cast<UnitAttr>());
  return nullptr;
}

} // namespace Circle
} // namespace mlir
