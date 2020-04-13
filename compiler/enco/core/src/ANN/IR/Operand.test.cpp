/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Operand.h"

#include <gtest/gtest.h>

TEST(ANN_IR_SCALAR_OPERAND, constructor)
{
  const ann::ScalarOperand operand;

  ASSERT_EQ(operand.dtype(), ann::DType::UNK);
  ASSERT_EQ(operand.weight(), nullptr);
}

TEST(ANN_IR_TENSOR_OPERAND, constructor)
{
  const nncc::core::ADT::tensor::Shape shape{1, 2};
  const ann::TensorOperand operand{shape};

  ASSERT_EQ(operand.dtype(), ann::DType::UNK);
  ASSERT_EQ(operand.weight(), nullptr);
  ASSERT_EQ(operand.shape(), shape);
}
