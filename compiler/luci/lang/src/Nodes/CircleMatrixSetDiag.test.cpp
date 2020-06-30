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

#include "luci/IR/Nodes/CircleMatrixSetDiag.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleMatrixSetDiagTest, constructor_P)
{
  luci::CircleMatrixSetDiag matrix_set_diag_node;

  ASSERT_EQ(luci::CircleDialect::get(), matrix_set_diag_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MATRIX_SET_DIAG, matrix_set_diag_node.opcode());

  ASSERT_EQ(nullptr, matrix_set_diag_node.input());
  ASSERT_EQ(nullptr, matrix_set_diag_node.diagonal());
}

TEST(CircleMatrixSetDiagTest, input_NEG)
{
  luci::CircleMatrixSetDiag matrix_set_diag_node;
  luci::CircleMatrixSetDiag node;

  matrix_set_diag_node.diagonal(&node);

  ASSERT_NE(nullptr, matrix_set_diag_node.input());
  ASSERT_NE(nullptr, matrix_set_diag_node.diagonal());

  matrix_set_diag_node.input(nullptr);
  matrix_set_diag_node.diagonal(nullptr);

  ASSERT_EQ(nullptr, matrix_set_diag_node.input());
  ASSERT_EQ(nullptr, matrix_set_diag_node.diagonal());
}

TEST(CircleMatrixSetDiagTest, arity_NEG)
{
  luci::CircleMatrixSetDiag matrix_set_diag_node;

  ASSERT_NO_THROW(matrix_set_diag_node.arg(0));
  ASSERT_NO_THROW(matrix_set_diag_node.arg(1));
  ASSERT_THROW(matrix_set_diag_node.arg(2), std::out_of_range);
}

TEST(CircleMatrixSetDiagTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleMatrixSetDiag matrix_set_diag_node;

  TestVisitor tv;
  ASSERT_THROW(matrix_set_diag_node.accept(&tv), std::exception);
}

TEST(CircleMatrixSetDiagTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleMatrixSetDiag matrix_set_diag_node;

  TestVisitor tv;
  ASSERT_THROW(matrix_set_diag_node.accept(&tv), std::exception);
}