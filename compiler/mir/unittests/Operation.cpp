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

#include "mir/Operation.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/SoftmaxOp.h"

#include <gtest/gtest.h>

using namespace mir;

TEST(Operation, ConnectionTest)
{

  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{}};
  auto op1 = new ops::InputOp(input_type);
  op1->setId(0);
  auto op2 = new ops::ReshapeOp(op1->getOutput(0), Shape{});
  op2->setId(1);

  ASSERT_EQ(op1, op2->getInput(0)->getNode());

  delete op1;
  delete op2;
}

TEST(Operation, InputOutputShapeTest)
{
  Shape input_shape{1, 2, 3};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  ops::InputOp input(input_type);
  ops::SoftmaxOp op(input.getOutput(0), 0);

  ASSERT_EQ(input_shape, input.getOutputShape(0));
  ASSERT_EQ(input_shape, op.getInputShape(0));
}

TEST(Operation, SoftmaxAxisTest)
{
  Shape input_shape{1, 2, 3};

  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  ops::InputOp input(input_type);

  ops::SoftmaxOp op_1(input.getOutput(0), 1);
  ASSERT_EQ(op_1.getAxis(), 1);

  ops::SoftmaxOp op_n1(input.getOutput(0), -1);
  ASSERT_EQ(op_n1.getAxis(), 2);

  ops::SoftmaxOp op_n3(input.getOutput(0), -3);
  ASSERT_EQ(op_n3.getAxis(), 0);
}

TEST(Operation, ConcatAxisTest)
{
  Shape in_shape{1, 2, 3};

  mir::TensorType in_type{mir::DataType::FLOAT32, in_shape};
  ops::InputOp input1(in_type), input2(in_type);

  ops::ConcatOp op_1({input1.getOutput(0), input2.getOutput(0)}, 1);
  ASSERT_EQ(op_1.getAxis(), 1);

  ops::ConcatOp op_n1({input1.getOutput(0), input2.getOutput(0)}, -1);
  ASSERT_EQ(op_n1.getAxis(), 2);

  ops::ConcatOp op_n3({input1.getOutput(0), input2.getOutput(0)}, -3);
  ASSERT_EQ(op_n3.getAxis(), 0);
}

TEST(Operation, OpNameTest)
{
#define HANDLE_OP(OpType, OpClass) ASSERT_EQ(getTypeName(Operation::Type::OpType), #OpType);
#include "mir/Operations.inc"
#undef HANDLE_OP
}
