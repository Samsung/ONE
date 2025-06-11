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

#include "ConvertHelper.h"

#include "circle-mlir/dialect/NameUtils.h"

#include <mlir/IR/Operation.h> // from @llvm-project
#include <mlir/IR/Matchers.h>  // from @llvm-project
#include <src/Dialect/ONNX/ONNXOps.hpp>
#include <src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp>

#include <cassert>
#include <vector>

namespace mlir
{
namespace Circle
{

std::string GetOperationName(mlir::Operation *op)
{
  assert(op != nullptr);

  mlir::Location opLoc = op->getLoc();
  auto name = mlir::GetNameFromLoc(opLoc);
  if (!name.empty())
    return name;

  // TOO remove this when not used anymore
  auto strattr = op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
  if (strattr)
    return strattr.str();

  // Use operator type as name if there is no name
  // TODO revise this to better implementation
  static uint64_t sequence = 1;
  auto seqstr = std::to_string(sequence);
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

namespace
{

template <typename T> bool ExtractConstantIntValues(mlir::Value &input, std::vector<T> &values)
{
  mlir::DenseElementsAttr dataAttr;

  if (auto constOp = dyn_cast_or_null<mlir::ONNXConstantOp>(input.getDefiningOp()))
  {
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp.getValueAttr());
    if (dataAttr == nullptr)
    {
      auto disValueAttr =
        mlir::dyn_cast_or_null<mlir::DisposableElementsAttr>(constOp.getValueAttr());
      if (disValueAttr)
        dataAttr = disValueAttr.toDenseElementsAttr();
    }
  }
  else if (auto constOp2 = dyn_cast_or_null<mlir::Circle::ConstOp>(input.getDefiningOp()))
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp2.getValueAttr());
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

template <typename T> bool ExtractConstantFloatValues(mlir::Value &input, std::vector<T> &values)
{
  mlir::DenseElementsAttr dataAttr;

  if (auto constOp = dyn_cast_or_null<mlir::ONNXConstantOp>(input.getDefiningOp()))
  {
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp.getValueAttr());
    if (dataAttr == nullptr)
    {
      auto disValueAttr =
        mlir::dyn_cast_or_null<mlir::DisposableElementsAttr>(constOp.getValueAttr());
      if (disValueAttr)
        dataAttr = disValueAttr.toDenseElementsAttr();
    }
  }
  else if (auto constOp2 = dyn_cast_or_null<mlir::Circle::ConstOp>(input.getDefiningOp()))
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp2.getValueAttr());
  else
    return false;

  if (dataAttr == nullptr)
    return false;

  auto valueIt = dataAttr.getValues<llvm::APFloat>().begin();
  auto valueEd = dataAttr.getValues<llvm::APFloat>().end();
  for (; valueIt != valueEd; ++valueIt)
  {
    T value = static_cast<T>((*valueIt).convertToFloat());
    values.push_back(value);
  }
  return true;
}

} // namespace

bool ExtractConstantValues(mlir::Value &input, std::vector<int32_t> &values)
{
  return ExtractConstantIntValues<int32_t>(input, values);
}

bool ExtractConstantValues(mlir::Value &input, std::vector<int64_t> &values)
{
  return ExtractConstantIntValues<int64_t>(input, values);
}

bool ExtractConstantValues(mlir::Value &input, std::vector<float> &values)
{
  return ExtractConstantFloatValues<float>(input, values);
}

namespace
{

template <typename T> void ExtractArrayAttrIntValues(mlir::ArrayAttr &array, std::vector<T> &values)
{
  for (int i = 0; i < array.size(); ++i)
  {
    auto v = GetIntValue<T>(array, i);
    values.push_back(v);
  }
}

} // namespace

void ExtractArrayAttrValues(mlir::ArrayAttr &array, std::vector<int32_t> &values)
{
  ExtractArrayAttrIntValues<int32_t>(array, values);
}

mlir::Value CreateNoValue(mlir::ConversionPatternRewriter &rewriter)
{
  return rewriter.create<NoValueOp>(rewriter.getUnknownLoc(), rewriter.getNoneType(),
                                    rewriter.getUnitAttr());
}

mlir::Value CreatePreTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                               mlir::Value &input)
{
  llvm::SmallVector<int32_t, 4> pre_vals{0, 2, 3, 1};
  mlir::Value pre_perm = rewriter.create<ConstOp>(opLoc, GetI32ElementsAttr(pre_vals, &rewriter));
  mlir::Value pre_tran = rewriter.create<TransposeOp>(opLoc, input, pre_perm);
  return pre_tran;
}

mlir::Value CreatePreTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Value &input,
                               std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/pre_tr/perm"));
  mlir::Location transLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/pre_tr"));
  llvm::SmallVector<int32_t, 4> pre_vals{0, 2, 3, 1};
  mlir::Value pre_perm =
    rewriter.create<ConstOp>(constLoc, GetI32ElementsAttr(pre_vals, &rewriter));
  return rewriter.create<TransposeOp>(transLoc, input, pre_perm);
}

mlir::Value CreateTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                            mlir::Value &input, llvm::SmallVector<int32_t, 4> &perm)
{
  mlir::Value perm_val = rewriter.create<ConstOp>(opLoc, GetI32ElementsAttr(perm, &rewriter));
  mlir::Value tran_val = rewriter.create<TransposeOp>(opLoc, input, perm_val);
  return tran_val;
}

mlir::Value CreateTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Value &input,
                            llvm::SmallVector<int32_t, 4> &perm, std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/tr/perm"));
  mlir::Location transLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/tr"));
  mlir::Value perm_val = rewriter.create<ConstOp>(constLoc, GetI32ElementsAttr(perm, &rewriter));
  return rewriter.create<TransposeOp>(transLoc, input, perm_val);
}

void ReplaceOpWithPostTranspose(mlir::ConversionPatternRewriter &rewriter, Operation *op,
                                mlir::Location &opLoc, mlir::Value &input, mlir::TypeRange type)
{
  llvm::SmallVector<int32_t, 4> post_vals{0, 3, 1, 2};
  mlir::Value post_perm = rewriter.create<ConstOp>(opLoc, GetI32ElementsAttr(post_vals, &rewriter));
  rewriter.replaceOpWithNewOp<TransposeOp>(op, type, input, post_perm);
}

mlir::Value ReplaceOpWithPostTranspose(mlir::ConversionPatternRewriter &rewriter, Operation *op,
                                       mlir::Value &input, mlir::TypeRange type, std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/post_tr/perm"));
  mlir::Location transLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/post_tr"));
  llvm::SmallVector<int32_t, 4> post_vals{0, 3, 1, 2};
  mlir::Value post_perm =
    rewriter.create<ConstOp>(constLoc, GetI32ElementsAttr(post_vals, &rewriter));
  mlir::Value post_tr = rewriter.create<TransposeOp>(transLoc, input, post_perm);
  rewriter.replaceOp(op, {post_tr});
  return post_tr;
}

mlir::Value ReplaceOpWithPostTranspose(mlir::PatternRewriter &rewriter, Operation *op,
                                       mlir::Value &input, mlir::TypeRange type, std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/post_tr/perm"));
  mlir::Location transLoc = mlir::NameLoc::get(rewriter.getStringAttr(name + "/post_tr"));
  llvm::SmallVector<int32_t, 4> post_vals{0, 3, 1, 2};
  mlir::Value post_perm =
    rewriter.create<ConstOp>(constLoc, GetI32ElementsAttr(post_vals, &rewriter));
  mlir::Value post_tr = rewriter.create<TransposeOp>(transLoc, input, post_perm);
  rewriter.replaceOp(op, {post_tr});
  return post_tr;
}

mlir::RankedTensorType GetChnLastType(mlir::RankedTensorType tensor_type)
{
  auto tensor_shape = tensor_type.getShape();
  // NCHW to NHWC
  auto to_nhwc = {tensor_shape[0], tensor_shape[2], tensor_shape[3], tensor_shape[1]};
  return mlir::RankedTensorType::get(to_nhwc, tensor_type.getElementType());
}

mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, float value,
                        const std::string &name)
{
  mlir::Type f32 = rewriter.getF32Type();
  mlir::RankedTensorType f32type = mlir::RankedTensorType::get({}, f32);
  llvm::SmallVector<float> values;
  values.push_back(value);
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return rewriter.create<ConstOp>(constLoc, mlir::DenseFPElementsAttr::get(f32type, values));
}

mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                        mlir::Value &reference, float value)
{
  auto rtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(reference.getType());
  if (not rtype)
    return {};
  if (not rtype.getElementType().isF32())
    return {};
  auto shape = rtype.getShape();
  if (shape.size() == 0)
    return {};

  // TODO revise to better value filling
  int64_t numElements = 1;
  for (size_t dim = 0; dim < shape.size(); ++dim)
    numElements = numElements * shape[dim];

  llvm::SmallVector<float> values;
  for (int64_t c = 0; c < numElements; ++c)
    values.push_back(value);

  return rewriter.create<ConstOp>(opLoc, mlir::DenseFPElementsAttr::get(rtype, values));
}

mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, mlir::Value &reference,
                        float value, const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return CreateConst(rewriter, constLoc, reference, value);
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           int64_t value)
{
  mlir::Type i32 = rewriter.getI32Type();
  mlir::RankedTensorType scalar_type = RankedTensorType::get({}, i32);
  auto avalue = static_cast<int32_t>(value);
  auto attr = mlir::DenseElementsAttr::get(scalar_type, {avalue});
  return rewriter.create<ConstOp>(opLoc, attr);
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, int64_t value,
                           const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  auto const_op = CreateI32Const(rewriter, constLoc, value);
  return const_op;
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           llvm::ArrayRef<int64_t> source)
{
  auto ssize = static_cast<int32_t>(source.size());
  std::vector<int32_t> values;
  for (int32_t i = 0; i < ssize; ++i)
    values.push_back(source[i]);
  mlir::Type i32 = rewriter.getI32Type();
  mlir::RankedTensorType ptype = RankedTensorType::get({ssize}, i32);
  return rewriter.create<ConstOp>(opLoc, DenseIntElementsAttr::get(ptype, values));
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter,
                           llvm::ArrayRef<int64_t> source, const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return CreateI32Const(rewriter, constLoc, source);
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           std::vector<int32_t> &source)
{
  auto num = static_cast<int64_t>(source.size());
  mlir::Type i32 = rewriter.getI32Type();
  mlir::RankedTensorType ptype = RankedTensorType::get({num}, i32);
  return rewriter.create<ConstOp>(opLoc, DenseIntElementsAttr::get(ptype, source));
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, std::vector<int32_t> &source,
                           const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return CreateI32Const(rewriter, constLoc, source);
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           mlir::Value &source)
{
  std::vector<int32_t> values;
  if (!ExtractConstantValues(source, values))
    return {};

  mlir::RankedTensorType stype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(source.getType());
  mlir::Type i32 = rewriter.getI32Type();
  mlir::RankedTensorType si16stype = RankedTensorType::get(stype.getShape(), i32);
  return rewriter.create<ConstOp>(opLoc, DenseIntElementsAttr::get(si16stype, values));
}

mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Value &source,
                           const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return CreateI32Const(rewriter, constLoc, source);
}

mlir::Value CreateConstBroadcastChn(mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location &opLoc, mlir::Value &reference,
                                    mlir::Value &source)
{
  // TODO support other dtypes
  // TODO support more source shape like 1xN, Nx1, ...
  // TODO revise to better form when known

  // check reference is rank4, F32
  auto rtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(reference.getType());
  auto rshape = rtype.getShape();
  if (not(rtype.getElementType().isF32() && rshape.size() == 4))
    return source;

  // check source is rank1, F32, same number of elements
  auto stype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(source.getType());
  auto sshape = stype.getShape();
  if (sshape.size() == rshape.size())
    return source;
  if (not(stype.getElementType().isF32() && sshape.size() == 1 && rshape[1] == sshape[0]))
    return source;

  int32_t C = rshape[1];

  mlir::DenseElementsAttr dataAttr;
  if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(source.getDefiningOp()))
  {
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp.getValueAttr());
    if (dataAttr == nullptr)
    {
      auto disValueAttr =
        mlir::dyn_cast_or_null<mlir::DisposableElementsAttr>(constOp.getValueAttr());
      if (disValueAttr)
        dataAttr = disValueAttr.toDenseElementsAttr();
    }
  }
  else if (auto constOp2 = dyn_cast<mlir::Circle::ConstOp>(source.getDefiningOp()))
    dataAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp2.getValueAttr());
  else
    return source;
  if (dataAttr == nullptr)
    return source;

  auto valueIt = dataAttr.getValues<llvm::APFloat>().begin();
  auto valueEd = dataAttr.getValues<llvm::APFloat>().end();
  llvm::SmallVector<float> values;
  for (; valueIt != valueEd; ++valueIt)
  {
    float val = (*valueIt).convertToFloat();
    values.push_back(val);
  }

  mlir::Type f32 = rewriter.getF32Type();
  mlir::RankedTensorType ttype = mlir::RankedTensorType::get({1, C, 1, 1}, f32);
  return rewriter.create<ConstOp>(opLoc, mlir::DenseFPElementsAttr::get(ttype, values));
}

mlir::Value CreateConstBroadcastChn(mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Value &reference, mlir::Value &source,
                                    const std::string &name)
{
  mlir::Location constLoc = mlir::NameLoc::get(rewriter.getStringAttr(name));
  return CreateConstBroadcastChn(rewriter, constLoc, reference, source);
}

bool GetPads(std::optional<::mlir::ArrayAttr> pads, std::vector<int32_t> &values)
{
  bool process = false;
  if (pads.has_value())
  {
    auto value = pads.value();
    // NOTE assert for not rank 4: this is for debug build to break the execution
    assert(value.size() == 4);
    // NOTE skip processing pads if not rank 4
    if (value.size() != 4)
      return process;
    for (int i = 0; i < value.size(); ++i)
    {
      auto v = GetIntValue<int32_t>(value, i);
      values.push_back(v);
      if (v)
        process = true;
    }
  }
  return process;
}

} // namespace Circle
} // namespace mlir
