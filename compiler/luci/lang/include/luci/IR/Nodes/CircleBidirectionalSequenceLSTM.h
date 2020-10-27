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

#ifndef __LUCI_IR_CIRCLEBIDIRECTIONALSEQUENCELSTM_H__
#define __LUCI_IR_CIRCLEBIDIRECTIONALSEQUENCELSTM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief BIDIRECTIONAL_SEQUENCE_LSTM in Circle
 */
class CircleBidirectionalSequenceLSTM final
    : public FixedArityNode<48, CircleNodeImpl<CircleOpcode::BIDIRECTIONAL_SEQUENCE_LSTM>>,
      public LuciNodeMixin<LuciNodeTrait::FusedActFunc>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *fw_input_to_input_weights(void) const { return at(1)->node(); }
  void fw_input_to_input_weights(loco::Node *node) { at(1)->node(node); }
  loco::Node *fw_input_to_forget_weights(void) const { return at(2)->node(); }
  void fw_input_to_forget_weights(loco::Node *node) { at(2)->node(node); }
  loco::Node *fw_input_to_cell_weights(void) const { return at(3)->node(); }
  void fw_input_to_cell_weights(loco::Node *node) { at(3)->node(node); }
  loco::Node *fw_input_to_output_weights(void) const { return at(4)->node(); }
  void fw_input_to_output_weights(loco::Node *node) { at(4)->node(node); }

  loco::Node *fw_recurrent_to_input_weights(void) const { return at(5)->node(); }
  void fw_recurrent_to_input_weights(loco::Node *node) { at(5)->node(node); }
  loco::Node *fw_recurrent_to_forget_weights(void) const { return at(6)->node(); }
  void fw_recurrent_to_forget_weights(loco::Node *node) { at(6)->node(node); }
  loco::Node *fw_recurrent_to_cell_weights(void) const { return at(7)->node(); }
  void fw_recurrent_to_cell_weights(loco::Node *node) { at(7)->node(node); }
  loco::Node *fw_recurrent_to_output_weights(void) const { return at(8)->node(); }
  void fw_recurrent_to_output_weights(loco::Node *node) { at(8)->node(node); }

  loco::Node *fw_cell_to_input_weights(void) const { return at(9)->node(); }
  void fw_cell_to_input_weights(loco::Node *node) { at(9)->node(node); }
  loco::Node *fw_cell_to_forget_weights(void) const { return at(10)->node(); }
  void fw_cell_to_forget_weights(loco::Node *node) { at(10)->node(node); }
  loco::Node *fw_cell_to_output_weights(void) const { return at(11)->node(); }
  void fw_cell_to_output_weights(loco::Node *node) { at(11)->node(node); }

  loco::Node *fw_input_gate_bias(void) const { return at(12)->node(); }
  void fw_input_gate_bias(loco::Node *node) { at(12)->node(node); }
  loco::Node *fw_forget_gate_bias(void) const { return at(13)->node(); }
  void fw_forget_gate_bias(loco::Node *node) { at(13)->node(node); }
  loco::Node *fw_cell_gate_bias(void) const { return at(14)->node(); }
  void fw_cell_gate_bias(loco::Node *node) { at(14)->node(node); }
  loco::Node *fw_output_gate_bias(void) const { return at(15)->node(); }
  void fw_output_gate_bias(loco::Node *node) { at(15)->node(node); }

  loco::Node *fw_projection_weights(void) const { return at(16)->node(); }
  void fw_projection_weights(loco::Node *node) { at(16)->node(node); }
  loco::Node *fw_projection_bias(void) const { return at(17)->node(); }
  void fw_projection_bias(loco::Node *node) { at(17)->node(node); }

  loco::Node *bw_input_to_input_weights(void) const { return at(18)->node(); }
  void bw_input_to_input_weights(loco::Node *node) { at(18)->node(node); }
  loco::Node *bw_input_to_forget_weights(void) const { return at(19)->node(); }
  void bw_input_to_forget_weights(loco::Node *node) { at(19)->node(node); }
  loco::Node *bw_input_to_cell_weights(void) const { return at(20)->node(); }
  void bw_input_to_cell_weights(loco::Node *node) { at(20)->node(node); }
  loco::Node *bw_input_to_output_weights(void) const { return at(21)->node(); }
  void bw_input_to_output_weights(loco::Node *node) { at(21)->node(node); }

  loco::Node *bw_recurrent_to_input_weights(void) const { return at(22)->node(); }
  void bw_recurrent_to_input_weights(loco::Node *node) { at(22)->node(node); }
  loco::Node *bw_recurrent_to_forget_weights(void) const { return at(23)->node(); }
  void bw_recurrent_to_forget_weights(loco::Node *node) { at(23)->node(node); }
  loco::Node *bw_recurrent_to_cell_weights(void) const { return at(24)->node(); }
  void bw_recurrent_to_cell_weights(loco::Node *node) { at(24)->node(node); }
  loco::Node *bw_recurrent_to_output_weights(void) const { return at(25)->node(); }
  void bw_recurrent_to_output_weights(loco::Node *node) { at(25)->node(node); }

  loco::Node *bw_cell_to_input_weights(void) const { return at(26)->node(); }
  void bw_cell_to_input_weights(loco::Node *node) { at(26)->node(node); }
  loco::Node *bw_cell_to_forget_weights(void) const { return at(27)->node(); }
  void bw_cell_to_forget_weights(loco::Node *node) { at(27)->node(node); }
  loco::Node *bw_cell_to_output_weights(void) const { return at(28)->node(); }
  void bw_cell_to_output_weights(loco::Node *node) { at(28)->node(node); }

  loco::Node *bw_input_gate_bias(void) const { return at(29)->node(); }
  void bw_input_gate_bias(loco::Node *node) { at(29)->node(node); }
  loco::Node *bw_forget_gate_bias(void) const { return at(30)->node(); }
  void bw_forget_gate_bias(loco::Node *node) { at(30)->node(node); }
  loco::Node *bw_cell_gate_bias(void) const { return at(31)->node(); }
  void bw_cell_gate_bias(loco::Node *node) { at(31)->node(node); }
  loco::Node *bw_output_gate_bias(void) const { return at(32)->node(); }
  void bw_output_gate_bias(loco::Node *node) { at(32)->node(node); }

  loco::Node *bw_projection_weights(void) const { return at(33)->node(); }
  void bw_projection_weights(loco::Node *node) { at(33)->node(node); }
  loco::Node *bw_projection_bias(void) const { return at(34)->node(); }
  void bw_projection_bias(loco::Node *node) { at(34)->node(node); }

  loco::Node *fw_activation_state(void) const { return at(35)->node(); }
  void fw_activation_state(loco::Node *node) { at(35)->node(node); }
  loco::Node *fw_cell_state(void) const { return at(36)->node(); }
  void fw_cell_state(loco::Node *node) { at(36)->node(node); }

  loco::Node *bw_activation_state(void) const { return at(37)->node(); }
  void bw_activation_state(loco::Node *node) { at(37)->node(node); }
  loco::Node *bw_cell_state(void) const { return at(38)->node(); }
  void bw_cell_state(loco::Node *node) { at(38)->node(node); }

  loco::Node *auxillary_input(void) const { return at(39)->node(); }
  void auxillary_input(loco::Node *node) { at(39)->node(node); }
  loco::Node *fw_auxillary_input_to_input_weights(void) const { return at(40)->node(); }
  void fw_auxillary_input_to_input_weights(loco::Node *node) { at(40)->node(node); }
  loco::Node *fw_auxillary_input_to_forget_weights(void) const { return at(41)->node(); }
  void fw_auxillary_input_to_forget_weights(loco::Node *node) { at(41)->node(node); }
  loco::Node *fw_auxillary_input_to_cell_weights(void) const { return at(42)->node(); }
  void fw_auxillary_input_to_cell_weights(loco::Node *node) { at(42)->node(node); }
  loco::Node *fw_auxillary_input_to_output_weights(void) const { return at(43)->node(); }
  void fw_auxillary_input_to_output_weights(loco::Node *node) { at(43)->node(node); }
  loco::Node *bw_auxillary_input_to_input_weights(void) const { return at(44)->node(); }
  void bw_auxillary_input_to_input_weights(loco::Node *node) { at(44)->node(node); }
  loco::Node *bw_auxillary_input_to_forget_weights(void) const { return at(45)->node(); }
  void bw_auxillary_input_to_forget_weights(loco::Node *node) { at(45)->node(node); }
  loco::Node *bw_auxillary_input_to_cell_weights(void) const { return at(46)->node(); }
  void bw_auxillary_input_to_cell_weights(loco::Node *node) { at(46)->node(node); }
  loco::Node *bw_auxillary_input_to_output_weights(void) const { return at(47)->node(); }
  void bw_auxillary_input_to_output_weights(loco::Node *node) { at(47)->node(node); }

public:
  float cell_clip(void) const { return _cell_clip; }
  void cell_clip(float cell_clip) { _cell_clip = cell_clip; }
  float proj_clip(void) const { return _proj_clip; }
  void proj_clip(float proj_clip) { _proj_clip = proj_clip; }
  bool merge_outputs(void) const { return _merge_outputs; }
  void merge_outputs(bool merge_outputs) { _merge_outputs = merge_outputs; }
  bool time_major(void) const { return _time_major; }
  void time_major(bool time_major) { _time_major = time_major; }
  bool asymmetric_quantize_inputs(void) const { return _asymmetric_quantize_inputs; }
  void asymmetric_quantize_inputs(bool asymmetric_quantize_inputs)
  {
    _asymmetric_quantize_inputs = asymmetric_quantize_inputs;
  }

private:
  float _cell_clip = 0.0f;
  float _proj_clip = 0.0f;
  bool _merge_outputs = false;
  bool _time_major = false;
  bool _asymmetric_quantize_inputs = false;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEBIDIRECTIONALSEQUENCELSTM_H__
