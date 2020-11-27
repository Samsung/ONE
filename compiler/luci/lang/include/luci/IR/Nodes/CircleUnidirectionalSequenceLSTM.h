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

#ifndef __LUCI_IR_CIRCLEUNIDIRECTIONALSEQUENCELSTM_H__
#define __LUCI_IR_CIRCLEUNIDIRECTIONALSEQUENCELSTM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief UNIDIRECTIONAL_SEQUENCE_LSTM in Circle
 */
class CircleUnidirectionalSequenceLSTM final
  : public FixedArityNode<24, CircleNodeImpl<CircleOpcode::UNIDIRECTIONAL_SEQUENCE_LSTM>>,
    public LuciNodeMixin<LuciNodeTrait::FusedActFunc>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *input_to_input_weights(void) const { return at(1)->node(); }
  void input_to_input_weights(loco::Node *node) { at(1)->node(node); }
  loco::Node *input_to_forget_weights(void) const { return at(2)->node(); }
  void input_to_forget_weights(loco::Node *node) { at(2)->node(node); }
  loco::Node *input_to_cell_weights(void) const { return at(3)->node(); }
  void input_to_cell_weights(loco::Node *node) { at(3)->node(node); }
  loco::Node *input_to_output_weights(void) const { return at(4)->node(); }
  void input_to_output_weights(loco::Node *node) { at(4)->node(node); }

  loco::Node *recurrent_to_input_weights(void) const { return at(5)->node(); }
  void recurrent_to_input_weights(loco::Node *node) { at(5)->node(node); }
  loco::Node *recurrent_to_forget_weights(void) const { return at(6)->node(); }
  void recurrent_to_forget_weights(loco::Node *node) { at(6)->node(node); }
  loco::Node *recurrent_to_cell_weights(void) const { return at(7)->node(); }
  void recurrent_to_cell_weights(loco::Node *node) { at(7)->node(node); }
  loco::Node *recurrent_to_output_weights(void) const { return at(8)->node(); }
  void recurrent_to_output_weights(loco::Node *node) { at(8)->node(node); }

  loco::Node *cell_to_input_weights(void) const { return at(9)->node(); }
  void cell_to_input_weights(loco::Node *node) { at(9)->node(node); }
  loco::Node *cell_to_forget_weights(void) const { return at(10)->node(); }
  void cell_to_forget_weights(loco::Node *node) { at(10)->node(node); }
  loco::Node *cell_to_output_weights(void) const { return at(11)->node(); }
  void cell_to_output_weights(loco::Node *node) { at(11)->node(node); }

  loco::Node *input_gate_bias(void) const { return at(12)->node(); }
  void input_gate_bias(loco::Node *node) { at(12)->node(node); }
  loco::Node *forget_gate_bias(void) const { return at(13)->node(); }
  void forget_gate_bias(loco::Node *node) { at(13)->node(node); }
  loco::Node *cell_gate_bias(void) const { return at(14)->node(); }
  void cell_gate_bias(loco::Node *node) { at(14)->node(node); }
  loco::Node *output_gate_bias(void) const { return at(15)->node(); }
  void output_gate_bias(loco::Node *node) { at(15)->node(node); }

  loco::Node *projection_weights(void) const { return at(16)->node(); }
  void projection_weights(loco::Node *node) { at(16)->node(node); }
  loco::Node *projection_bias(void) const { return at(17)->node(); }
  void projection_bias(loco::Node *node) { at(17)->node(node); }

  loco::Node *activation_state(void) const { return at(18)->node(); }
  void activation_state(loco::Node *node) { at(18)->node(node); }
  loco::Node *cell_state(void) const { return at(19)->node(); }
  void cell_state(loco::Node *node) { at(19)->node(node); }

  loco::Node *input_layer_norm_coefficients(void) const { return at(20)->node(); }
  void input_layer_norm_coefficients(loco::Node *node) { at(20)->node(node); }
  loco::Node *forget_layer_norm_coefficients(void) const { return at(21)->node(); }
  void forget_layer_norm_coefficients(loco::Node *node) { at(21)->node(node); }
  loco::Node *cell_layer_norm_coefficients(void) const { return at(22)->node(); }
  void cell_layer_norm_coefficients(loco::Node *node) { at(22)->node(node); }
  loco::Node *output_layer_norm_coefficients(void) const { return at(23)->node(); }
  void output_layer_norm_coefficients(loco::Node *node) { at(23)->node(node); }

public:
  float cell_clip(void) const { return _cell_clip; }
  void cell_clip(float cell_clip) { _cell_clip = cell_clip; }
  float proj_clip(void) const { return _proj_clip; }
  void proj_clip(float proj_clip) { _proj_clip = proj_clip; }
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
  bool _time_major = false;
  bool _asymmetric_quantize_inputs = false;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEUNIDIRECTIONALSEQUENCELSTM_H__
