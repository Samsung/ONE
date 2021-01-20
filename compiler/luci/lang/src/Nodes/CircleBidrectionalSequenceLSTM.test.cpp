/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleBidirectionalSequenceLSTM.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleBidirectionalSequenceLSTMTest, constructor_P)
{
  luci::CircleBidirectionalSequenceLSTM trc_node;

  ASSERT_EQ(luci::CircleDialect::get(), trc_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::BIDIRECTIONAL_SEQUENCE_LSTM, trc_node.opcode());

  ASSERT_EQ(nullptr, trc_node.input());

  ASSERT_EQ(nullptr, trc_node.fw_input_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.fw_input_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.fw_input_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.fw_input_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.fw_recurrent_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.fw_recurrent_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.fw_recurrent_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.fw_recurrent_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.fw_cell_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.fw_cell_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.fw_cell_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.fw_input_gate_bias());
  ASSERT_EQ(nullptr, trc_node.fw_forget_gate_bias());
  ASSERT_EQ(nullptr, trc_node.fw_cell_gate_bias());
  ASSERT_EQ(nullptr, trc_node.fw_output_gate_bias());

  ASSERT_EQ(nullptr, trc_node.fw_projection_weights());
  ASSERT_EQ(nullptr, trc_node.fw_projection_bias());

  ASSERT_EQ(nullptr, trc_node.bw_input_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.bw_input_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.bw_input_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.bw_input_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.bw_recurrent_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.bw_recurrent_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.bw_recurrent_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.bw_recurrent_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.bw_cell_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.bw_cell_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.bw_cell_to_output_weights());

  ASSERT_EQ(nullptr, trc_node.bw_input_gate_bias());
  ASSERT_EQ(nullptr, trc_node.bw_forget_gate_bias());
  ASSERT_EQ(nullptr, trc_node.bw_cell_gate_bias());
  ASSERT_EQ(nullptr, trc_node.bw_output_gate_bias());

  ASSERT_EQ(nullptr, trc_node.bw_projection_weights());
  ASSERT_EQ(nullptr, trc_node.bw_projection_bias());

  ASSERT_EQ(nullptr, trc_node.fw_activation_state());
  ASSERT_EQ(nullptr, trc_node.fw_cell_state());
  ASSERT_EQ(nullptr, trc_node.bw_activation_state());
  ASSERT_EQ(nullptr, trc_node.bw_cell_state());

  ASSERT_EQ(nullptr, trc_node.auxillary_input());
  ASSERT_EQ(nullptr, trc_node.fw_auxillary_input_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.fw_auxillary_input_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.fw_auxillary_input_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.fw_auxillary_input_to_output_weights());
  ASSERT_EQ(nullptr, trc_node.bw_auxillary_input_to_input_weights());
  ASSERT_EQ(nullptr, trc_node.bw_auxillary_input_to_forget_weights());
  ASSERT_EQ(nullptr, trc_node.bw_auxillary_input_to_cell_weights());
  ASSERT_EQ(nullptr, trc_node.bw_auxillary_input_to_output_weights());

  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, trc_node.fusedActivationFunction());
  ASSERT_EQ(0.f, trc_node.cell_clip());
  ASSERT_EQ(0.f, trc_node.proj_clip());
  ASSERT_EQ(false, trc_node.merge_outputs());
  ASSERT_EQ(false, trc_node.time_major());
  ASSERT_EQ(false, trc_node.asymmetric_quantize_inputs());
}

TEST(CircleBidirectionalSequenceLSTMTest, arity_NEG)
{
  luci::CircleBidirectionalSequenceLSTM trc_node;

  ASSERT_NO_THROW(trc_node.arg(36));
  ASSERT_THROW(trc_node.arg(48), std::out_of_range);
}

TEST(CircleBidirectionalSequenceLSTMTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleBidirectionalSequenceLSTM trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}

TEST(CircleBidirectionalSequenceLSTMTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleBidirectionalSequenceLSTM trc_node;

  TestVisitor tv;
  ASSERT_THROW(trc_node.accept(&tv), std::exception);
}
