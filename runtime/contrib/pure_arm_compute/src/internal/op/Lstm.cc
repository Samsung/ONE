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

#include "internal/op/Lstm.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace LSTM
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace LSTM
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace LSTM
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 23 && outputCount == 4);

  scratch_buffer_index = outputs[0];
  output_state_out_index = outputs[1];
  cell_state_out_index = outputs[2];
  output_index = outputs[3];

  input_index = inputs[0];
  input_to_input_weights_index = inputs[1];
  input_to_forget_weights_index = inputs[2];
  input_to_cell_weights_index = inputs[3];
  input_to_output_weights_index = inputs[4];
  recurrent_to_input_weights_index = inputs[5];
  recurrent_to_forget_weights_index = inputs[6];
  recurrent_to_cell_weights_index = inputs[7];
  recurrent_to_output_weights_index = inputs[8];
  cell_to_input_weights_index = inputs[9];
  cell_to_forget_weights_index = inputs[10];
  cell_to_output_weights_index = inputs[11];
  input_gate_bias_index = inputs[12];
  forget_gate_bias_index = inputs[13];
  cell_bias_index = inputs[14];
  output_gate_bias_index = inputs[15];
  projection_weights_index = inputs[16];
  projection_bias_index = inputs[17];
  output_state_in_index = inputs[18];
  cell_state_in_index = inputs[19];
  activation_index = inputs[20];
  cell_threshold_index = inputs[21];
  projection_threshold_index = inputs[22];
}

} // namespace LSTM
} // namespace op
} // namespace tflite
} // namespace internal
