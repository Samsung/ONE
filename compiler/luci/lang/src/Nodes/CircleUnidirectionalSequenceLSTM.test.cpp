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

#include "luci/IR/Nodes/CircleUnidirectionalSequenceLSTM.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleUnidirectionalSequenceLSTMTest, constructor_P)
{
  luci::CircleUnidirectionalSequenceLSTM trc_node;

  ASSERT_EQ(luci::CircleDialect::get(), trc_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::UNIDIRECTIONAL_SEQUENCE_LSTM, trc_node.opcode());

  ASSERT_EQ(nullptr, trc_node.input());

  ASSERT_EQ(nullptr, trc_node.input_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.input_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.input_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.input_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.recurrent_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.recurrent_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.recurrent_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.recurrent_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.cell_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.cell_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.cell_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.input_gate_bias());
  ASSERT_EQ(nullptr, trc_node.forget_gate_bias());
  ASSERT_EQ(nullptr, trc_node.cell_gate_bias());
  ASSERT_EQ(nullptr, trc_node.output_gate_bias());

  ASSERT_EQ(nullptr, trc_node.projection_weights());
  ASSERT_EQ(nullptr, trc_node.projection_bias());

  ASSERT_EQ(nullptr, trc_node.activation_state());
  ASSERT_EQ(nullptr, trc_node.cell_state());

  ASSERT_EQ(nullptr, trc_node.input_layer_norm_coefficients());
  ASSERT_EQ(nullptr, trc_node.forget_layer_norm_coefficients());
  ASSERT_EQ(nullptr, trc_node.cell_layer_norm_coefficients());
  ASSERT_EQ(nullptr, trc_node.output_layer_norm_coefficients());

  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, trc_node.fusedActivationFunction());
  ASSERT_EQ(0.f, trc_node.cell_clip());
  ASSERT_EQ(0.f, trc_node.proj_clip());
  ASSERT_EQ(false, trc_node.time_major());
}

TEST(CircleUnidirectionalSequenceLSTMTest, arity_NEG)
{
  luci::CircleUnidirectionalSequenceLSTM trc_node;

  ASSERT_NO_THROW(trc_node.arg(20));
  ASSERT_THROW(trc_node.arg(24), std::out_of_range);
}

TEST(CircleUnidirectionalSequenceLSTMTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleUnidirectionalSequenceLSTM trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}

TEST(CircleUnidirectionalSequenceLSTMTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleUnidirectionalSequenceLSTM trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}
