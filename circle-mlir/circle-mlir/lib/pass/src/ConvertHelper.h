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

#ifndef __CIRCLE_MLIR_PASS_CONVERT_HELPER_H__
#define __CIRCLE_MLIR_PASS_CONVERT_HELPER_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Transforms/DialectConversion.h>

#include <string>

namespace mlir
{
namespace Circle
{

inline const char ACT_NONE[]{"NONE"};
inline const char ACT_RELU[]{"RELU"};
inline const char ACT_RELU6[]{"RELU6"};
inline const char ACT_TANH[]{"TANH"};

// Get name of the Op
std::string GetOperationName(mlir::Operation *op);

// Returns 1D 1-bit dense elements attribute with the given values.
DenseIntElementsAttr GetI1ElementsAttr(ArrayRef<bool> values, Builder *builder);

// Returns 1D 32-bit dense elements attribute with the given values.
DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values, Builder *builder);

bool ExtractConstantValues(mlir::Value &input, std::vector<int32_t> &values);
bool ExtractConstantValues(mlir::Value &input, std::vector<int64_t> &values);
bool ExtractConstantValues(mlir::Value &input, std::vector<float> &values);

void ExtractArrayAttrValues(mlir::ArrayAttr &array, std::vector<int32_t> &values);

// Create NoValueOp, used for No Bias
mlir::Value CreateNoValue(mlir::ConversionPatternRewriter &rewriter);

// Create and return TransposeOp for channel first(NCHW) to channel last(NHWC)
// channel first: NCHW, channel last: NHWC
mlir::Value CreatePreTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                               mlir::Value &input);
mlir::Value CreatePreTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Value &input,
                               const std::string &name);

// Create and return TransposeOp with permutation value
mlir::Value CreateTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                            mlir::Value &input, llvm::SmallVector<int32_t, 4> &perm);
mlir::Value CreateTranspose(mlir::ConversionPatternRewriter &rewriter, mlir::Value &input,
                            llvm::SmallVector<int32_t, 4> &perm, const std::string &name);

// Create and replace with TransposeOp for channel last(NHWC) to channel first(NCHW)
void ReplaceOpWithPostTranspose(mlir::ConversionPatternRewriter &rewriter, Operation *op,
                                mlir::Location &opLoc, mlir::Value &input, mlir::TypeRange type);
mlir::Value ReplaceOpWithPostTranspose(mlir::ConversionPatternRewriter &rewriter, Operation *op,
                                       mlir::Value &input, mlir::TypeRange type,
                                       const std::string &name);
mlir::Value ReplaceOpWithPostTranspose(mlir::PatternRewriter &rewriter, Operation *op,
                                       mlir::Value &input, mlir::TypeRange type,
                                       const std::string &name);

// Get output type of op with channel last order
mlir::RankedTensorType GetChnLastType(mlir::RankedTensorType tensor_type);

// Create ConstOp with scalar type and float value
mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, float value,
                        const std::string &name);
// Create ConstOp with type 'type' and 'value' values
mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                        mlir::RankedTensorType &type, float value);
// Create ConstOp with type 'reference' and 'value' values
mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                        mlir::Value &reference, float value);
mlir::Value CreateConst(mlir::ConversionPatternRewriter &rewriter, mlir::Value &reference,
                        float value, const std::string &name);

// Create scalar ConstOp with value
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           int64_t value);
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, int64_t value,
                           const std::string &name);

// Create int32_t 1D ConstOp with ArrayRef
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           llvm::ArrayRef<int64_t> values);
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter,
                           llvm::ArrayRef<int64_t> source, const std::string &name);

// Create int32_t 1D ConstOp with std:vector
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           std::vector<int32_t> &source);
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, std::vector<int32_t> &source,
                           const std::string &name);

// Create int32_t ConstOp from int32_t/int64_t
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Location &opLoc,
                           mlir::Value &source);
mlir::Value CreateI32Const(mlir::ConversionPatternRewriter &rewriter, mlir::Value &source,
                           const std::string &name);

// Create ConstOp with broadcast applied to rank of 'reference' with values from 'source'
// - 'reference' is rank-4, NCHW format
// - 'source' is vector with C elemnts
// - return rank-4 with shape 1xCx1x1
// - only F32 is supported for now
// - return source if any condition does not match
mlir::Value CreateConstBroadcastChn(mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location &opLoc, mlir::Value &reference,
                                    mlir::Value &source);
mlir::Value CreateConstBroadcastChn(mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Value &reference, mlir::Value &source,
                                    const std::string &name);

// Get integer value of array[index]
template <typename TYPE> TYPE GetIntValue(mlir::ArrayAttr array, int index)
{
  return static_cast<TYPE>(mlir::cast<IntegerAttr>(array.getValue()[index]).getInt());
}

// Get value from pads if available as return true
// if false, we do not need to process pads value
bool GetPads(std::optional<::mlir::ArrayAttr> pads, std::vector<int32_t> &values);

#define CHECK_VALID_RANK_2(VALUE) \
  do                              \
  {                               \
    if (not VALUE)                \
      return mlir::failure();     \
    if (VALUE.getRank() != 2)     \
      return mlir::failure();     \
  } while (0)

#define CHECK_VALID_RANK_4(VALUE) \
  do                              \
  {                               \
    if (not VALUE)                \
      return mlir::failure();     \
    if (VALUE.getRank() != 4)     \
      return mlir::failure();     \
  } while (0)

#define CHECK_VALID_RANK_2_4(VALUE)                      \
  do                                                     \
  {                                                      \
    if (not VALUE)                                       \
      return mlir::failure();                            \
    if (!(VALUE.getRank() == 2 || VALUE.getRank() == 4)) \
      return mlir::failure();                            \
  } while (0)

#define CHECK_VALID_RANK_3_4(VALUE)                      \
  do                                                     \
  {                                                      \
    if (not VALUE)                                       \
      return mlir::failure();                            \
    if (!(VALUE.getRank() == 3 || VALUE.getRank() == 4)) \
      return mlir::failure();                            \
  } while (0)

#define CHECK_VALID_RANK_ATLEAST(VALUE, NUM) \
  do                                         \
  {                                          \
    if (not VALUE)                           \
      return mlir::failure();                \
    if (VALUE.getRank() < NUM)               \
      return mlir::failure();                \
  } while (0)

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_CONVERT_HELPER_H__
