/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include "ir/Operands.h"

TEST(graph_operand_Set, neg_set_test)
{
  onert::ir::Operands set;

  onert::ir::Shape shape0{1, 2, 3};

  onert::ir::Shape shape1(4);
  shape1.dim(0) = 10;
  shape1.dim(1) = 20;
  shape1.dim(2) = 30;
  shape1.dim(3) = 40;

  onert::ir::TypeInfo type{onert::ir::DataType::INT32};

  set.emplace(shape0, type);
  set.emplace(shape1, type);

  ASSERT_EQ(set.exist(onert::ir::OperandIndex{0u}), true);
  ASSERT_EQ(set.exist(onert::ir::OperandIndex{1u}), true);
  ASSERT_EQ(set.exist(onert::ir::OperandIndex{2u}), false);

  ASSERT_EQ(set.at(onert::ir::OperandIndex{0u}).shape().dim(0), 1);
  ASSERT_EQ(set.at(onert::ir::OperandIndex{0u}).shape().dim(1), 2);
  ASSERT_EQ(set.at(onert::ir::OperandIndex{0u}).shape().dim(2), 3);
}
