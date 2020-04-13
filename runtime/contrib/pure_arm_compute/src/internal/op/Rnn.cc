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

#include "internal/op/Rnn.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace RNN
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace RNN
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace RNN
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 6 && outputCount == 2);

  output_index = outputs[0];
  hidden_state_out_index = outputs[1];

  input_index = inputs[0];
  weights_index = inputs[1];
  recurrent_weights_index = inputs[2];
  bias_index = inputs[3];
  hidden_state_in_index = inputs[4];
  fused_activation_index = inputs[5];
}

} // namespace RNN
} // namespace op
} // namespace tflite
} // namespace internal
