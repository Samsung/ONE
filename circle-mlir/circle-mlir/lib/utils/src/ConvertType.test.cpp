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

#include "circle-mlir/utils/ConvertType.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <gtest/gtest.h>

using namespace circle;

class ConvertTypeTest : public ::testing::Test
{
protected:
  mlir::MLIRContext _context;
};

TEST_F(ConvertTypeTest, ConvertTypeToTensorType)
{
  auto builder = mlir::Builder(&_context);

  auto tF16 = builder.getF16Type();
  ASSERT_EQ(ConvertTypeToTensorType(tF16), circle::TensorType_FLOAT16);

  auto tF32 = builder.getF32Type();
  ASSERT_EQ(ConvertTypeToTensorType(tF32), circle::TensorType_FLOAT32);

  auto tF64 = builder.getF64Type();
  ASSERT_EQ(ConvertTypeToTensorType(tF64), circle::TensorType_FLOAT64);

  mlir::ComplexType cpxF64 = mlir::ComplexType::get(tF32);
  ASSERT_EQ(ConvertTypeToTensorType(cpxF64), circle::TensorType_COMPLEX64);

  mlir::ComplexType cpxF128 = mlir::ComplexType::get(tF64);
  ASSERT_EQ(ConvertTypeToTensorType(cpxF128), circle::TensorType_COMPLEX128);

  auto tI1 = builder.getI1Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI1), circle::TensorType_BOOL);

  auto tI4 = builder.getI4Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI4), circle::TensorType_INT4);

  auto tU4 = builder.getIntegerType(4, false);
  ASSERT_EQ(ConvertTypeToTensorType(tU4), circle::TensorType_UINT4);

  auto tI8 = builder.getI8Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI8), circle::TensorType_INT8);

  auto tU8 = builder.getIntegerType(8, false);
  ASSERT_EQ(ConvertTypeToTensorType(tU8), circle::TensorType_UINT8);

  auto tI16 = builder.getI16Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI16), circle::TensorType_INT16);

  auto tI32 = builder.getI32Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI32), circle::TensorType_INT32);

  auto tI64 = builder.getI64Type();
  ASSERT_EQ(ConvertTypeToTensorType(tI64), circle::TensorType_INT64);

  auto tU64 = builder.getIntegerType(64, false);
  ASSERT_EQ(ConvertTypeToTensorType(tU64), circle::TensorType_UINT64);
}

TEST_F(ConvertTypeTest, ConvertTypeToTensorType_NEG)
{
  auto builder = mlir::Builder(&_context);

  // there is no Complex32(F16,F16)
  auto tF16 = builder.getF16Type();
  mlir::ComplexType cpxF32 = mlir::ComplexType::get(tF16);
  EXPECT_THROW(ConvertTypeToTensorType(cpxF32), std::runtime_error);

  // strange width
  auto t19 = builder.getIntegerType(19, false);
  EXPECT_THROW(ConvertTypeToTensorType(t19), std::runtime_error);
}

TEST_F(ConvertTypeTest, ConvertElementType)
{
  auto builder = mlir::Builder(&_context);

  circle::TensorType type = circle::TensorType_FLOAT16;
  ASSERT_TRUE(ConvertElementType(type, builder).isF16());

  type = circle::TensorType_FLOAT32;
  ASSERT_TRUE(ConvertElementType(type, builder).isF32());

  type = circle::TensorType_FLOAT64;
  ASSERT_TRUE(ConvertElementType(type, builder).isF64());

  type = circle::TensorType_BOOL;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(1));

  type = circle::TensorType_INT4;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(4));

  type = circle::TensorType_UINT4;
  ASSERT_TRUE(ConvertElementType(type, builder).isUnsignedInteger(4));

  type = circle::TensorType_INT8;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(8));

  type = circle::TensorType_UINT8;
  ASSERT_TRUE(ConvertElementType(type, builder).isUnsignedInteger(8));

  type = circle::TensorType_INT16;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(16));

  type = circle::TensorType_UINT16;
  ASSERT_TRUE(ConvertElementType(type, builder).isUnsignedInteger(16));

  type = circle::TensorType_INT32;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(32));

  type = circle::TensorType_UINT32;
  ASSERT_TRUE(ConvertElementType(type, builder).isUnsignedInteger(32));

  type = circle::TensorType_INT64;
  ASSERT_TRUE(ConvertElementType(type, builder).isInteger(64));

  type = circle::TensorType_UINT64;
  ASSERT_TRUE(ConvertElementType(type, builder).isUnsignedInteger(64));
}

TEST_F(ConvertTypeTest, ConvertElementType_NEG)
{
  auto builder = mlir::Builder(&_context);

  circle::TensorType type = circle::TensorType_FLOAT32;
  ASSERT_FALSE(ConvertElementType(type, builder).isF16());
}
