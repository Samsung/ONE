/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleMatrixDiag.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMatrixDiagTest, constructor_P)
{
  luci::CircleMatrixDiag matrix_diag_node;

  ASSERT_EQ(luci::CircleDialect::get(), matrix_diag_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MATRIX_DIAG, matrix_diag_node.opcode());

  ASSERT_EQ(nullptr, matrix_diag_node.diagonal());
}

TEST(CircleMatrixDiagTest, input_NEG)
{
  luci::CircleMatrixDiag matrix_diag_node;
  luci::CircleMatrixDiag node;

  matrix_diag_node.diagonal(&node);

  ASSERT_NE(nullptr, matrix_diag_node.diagonal());

  matrix_diag_node.diagonal(nullptr);

  ASSERT_EQ(nullptr, matrix_diag_node.diagonal());
}

TEST(CircleMatrixDiagTest, arity_NEG)
{
  luci::CircleMatrixDiag matrix_diag_node;

  ASSERT_NO_THROW(matrix_diag_node.arg(0));
  ASSERT_THROW(matrix_diag_node.arg(1), std::out_of_range);
}

TEST(CircleMatrixDiagTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMatrixDiag matrix_diag_node;

  TestVisitor tv;
  ASSERT_THROW(matrix_diag_node.accept(&tv), std::exception);
}

TEST(CircleMatrixDiagTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMatrixDiag matrix_diag_node;

  TestVisitor tv;
  ASSERT_THROW(matrix_diag_node.accept(&tv), std::exception);
}
