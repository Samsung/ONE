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

#include "BidirectionalSequenceLSTM.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpBidirectionalSequenceLSTM::filler(const tflite::Operator *op, TFliteImport *import,
                                               tflchef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  assert(inputs.size() == 48);

  for (int32_t i = 0; i < inputs.size(); i++)
  {
    // Except for Input 0, 35, 36, 37 and 38.
    // Each Input mean Input Tensor, ActivationState Tensor (forward and backward), and CellState
    // Tensor (forward and backward).
    // This could be updated from previous input or User Given data, so This could not be Const
    if (i == 0 || i == 35 || i == 36 || i == 37 || i == 38)
      continue;
    if (inputs[i] != -1)
      fill_tensor_to_import(inputs[i], import);
  }
}

tflchef::Operation *TFliteOpBidirectionalSequenceLSTM::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_BidirectionalSequenceLSTMOptions();
  assert(op_params != nullptr);

  operation->set_type("BidirectionalSequenceLSTM");

  auto op_options = operation->mutable_bidirectional_sequence_lstm_options();

  op_options->set_activation(as_tflchef_activation(op_params->fused_activation_function()));
  op_options->set_cell_clip(op_params->cell_clip());
  op_options->set_proj_clip(op_params->proj_clip());
  op_options->set_time_major(op_params->time_major());
  op_options->set_asymmetric_quantize_inputs(op_params->asymmetric_quantize_inputs());
  op_options->set_merge_outputs(op_params->merge_outputs());

  return operation;
}

} // namespace tflchef
