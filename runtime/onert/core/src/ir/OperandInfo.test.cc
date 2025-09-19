/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/OperandInfo.h"
#include "util/Exceptions.h"

#include <gtest/gtest.h>

using namespace onert::ir;

TEST(ir_OperandInfo, total_size)
{
  auto info = OperandInfo::createStaticInfo(Shape{1, 2, 3}, TypeInfo{DataType::FLOAT32});
  EXPECT_EQ(info.total_size(), 24);

  info = OperandInfo::createStaticInfo(Shape{1, 2, 3}, TypeInfo{DataType::QUANT_INT8_SYMM});
  EXPECT_EQ(info.total_size(), 6);

  // Block quantization type operand
  info = OperandInfo::createStaticInfo(Shape{1, 4, 32}, TypeInfo{DataType::QUANT_GGML_Q4_0});
  EXPECT_EQ(info.total_size(), 18 * 4);
}

// Unsupported type
TEST(ir_OperandInfo, neg_total_size_type)
{
  auto info = OperandInfo::createStaticInfo(Shape{1, 2, 3}, TypeInfo{DataType{-1}});
  EXPECT_THROW(info.total_size(), onert::UnsupportedDataTypeException);
}

// Unsupported shape
TEST(ir_OperandInfo, neg_total_size_dimension)
{
  // Unspecified shape
  auto info = OperandInfo::createStaticInfo(Shape{1, -1, 3}, TypeInfo{DataType::FLOAT32});
  EXPECT_THROW(info.total_size(), std::runtime_error);

  // Block quantization operand
  info = OperandInfo::createStaticInfo(Shape{1, 2, 3}, TypeInfo{DataType::QUANT_GGML_Q4_0});
  EXPECT_THROW(info.total_size(), std::runtime_error);
  info = OperandInfo::createStaticInfo(Shape{1, 2, 5}, TypeInfo{DataType::QUANT_GGML_Q8_0});
  EXPECT_THROW(info.total_size(), std::runtime_error);
}
