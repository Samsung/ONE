/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAINING_OPS_LSTMLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_LSTMLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"
#include <ir/InternalType.h>
#include <ir/operation/LSTM.h>
#include <exec/IFunction.h>

namespace nnfw
{
namespace cker
{
class FCTempArena;
}
} // namespace nnfw

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

// TODO Support LSTM, BiDirectionalSequenceLSTM
class LSTMLayer : public ::onert::exec::IFunction
{
public:
  LSTMLayer() = default;

public:
  void LSTMFloat();

  void configure(
    const IPortableTensor *input, const IPortableTensor *input_to_input_weights,
    const IPortableTensor *input_to_forget_weights, const IPortableTensor *input_to_cell_weights,
    const IPortableTensor *input_to_output_weights,
    const IPortableTensor *recurrent_to_input_weights,
    const IPortableTensor *recurrent_to_forget_weights,
    const IPortableTensor *recurrent_to_cell_weights,
    const IPortableTensor *recurrent_to_output_weights,
    const IPortableTensor *cell_to_input_weights, const IPortableTensor *cell_to_forget_weights,
    const IPortableTensor *cell_to_output_weights, const IPortableTensor *input_layer_norm_weights,
    const IPortableTensor *forget_layer_norm_weights,
    const IPortableTensor *cell_layer_norm_weights,
    const IPortableTensor *output_layer_norm_weights, const IPortableTensor *aux_input,
    const IPortableTensor *aux_input_to_input_weights,
    const IPortableTensor *aux_input_to_forget_weights,
    const IPortableTensor *aux_input_to_cell_weights,
    const IPortableTensor *aux_input_to_output_weights, const IPortableTensor *input_gate_bias,
    const IPortableTensor *forget_gate_bias, const IPortableTensor *cell_gate_bias,
    const IPortableTensor *output_gate_bias, const IPortableTensor *projection_weights,
    const IPortableTensor *projection_bias, const IPortableTensor *output_state_in,
    const IPortableTensor *cell_state_in, const ir::operation::LSTM::Param &params,
    bool forward_sequence, bool time_major, int32_t output_offset, IPortableTensor *scratch_buffer,
    IPortableTensor *output_state, IPortableTensor *cell_state, IPortableTensor *output,
    bool has_output_state_data, bool has_cell_state_data);

  void run() override;

private:
  const IPortableTensor *_input{nullptr};
  const IPortableTensor *_input_to_input_weights{nullptr};
  const IPortableTensor *_input_to_forget_weights{nullptr};
  const IPortableTensor *_input_to_cell_weights{nullptr};
  const IPortableTensor *_input_to_output_weights{nullptr};
  const IPortableTensor *_recurrent_to_input_weights{nullptr};
  const IPortableTensor *_recurrent_to_forget_weights{nullptr};
  const IPortableTensor *_recurrent_to_cell_weights{nullptr};
  const IPortableTensor *_recurrent_to_output_weights{nullptr};
  const IPortableTensor *_cell_to_input_weights{nullptr};
  const IPortableTensor *_cell_to_forget_weights{nullptr};
  const IPortableTensor *_cell_to_output_weights{nullptr};
  const IPortableTensor *_input_layer_norm_coefficients{nullptr};
  const IPortableTensor *_forget_layer_norm_coefficients{nullptr};
  const IPortableTensor *_cell_layer_norm_coefficients{nullptr};
  const IPortableTensor *_output_layer_norm_coefficients{nullptr};
  const IPortableTensor *_aux_input{nullptr};
  const IPortableTensor *_aux_input_to_input_weights{nullptr};
  const IPortableTensor *_aux_input_to_forget_weights{nullptr};
  const IPortableTensor *_aux_input_to_cell_weights{nullptr};
  const IPortableTensor *_aux_input_to_output_weights{nullptr};
  const IPortableTensor *_input_gate_bias{nullptr};
  const IPortableTensor *_forget_gate_bias{nullptr};
  const IPortableTensor *_cell_gate_bias{nullptr};
  const IPortableTensor *_output_gate_bias{nullptr};
  const IPortableTensor *_projection_weights{nullptr};
  const IPortableTensor *_projection_bias{nullptr};
  const IPortableTensor *_output_state_in{nullptr};
  const IPortableTensor *_cell_state_in{nullptr};
  IPortableTensor *_scratch_buffer{nullptr};
  IPortableTensor *_output_state{nullptr};
  IPortableTensor *_cell_state{nullptr};
  IPortableTensor *_output{nullptr};
  std::vector<uint8_t> _scratch_vec{};
  std::vector<uint8_t> _output_state_vec{};
  std::vector<uint8_t> _cell_state_vec{};
  ir::operation::LSTM::Param _params{};
  bool _forward_sequence{true};
  bool _time_major{true};
  int32_t _output_offset{0};
  bool _has_output_state_data{false};
  bool _has_cell_state_data{false};
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_LSTMLAYER_H__
