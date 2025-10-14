/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "LSTMLayer.h"

#include "OperationUtils.h"

#include <cker/operation/LSTM.h>

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::LSTM &node)
{
  const auto scratch_buffer_index{
    node.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};
  const auto output_state_out_index{
    node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};
  const auto cell_state_out_index{
    node.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};
  const auto output_index{node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};

  const auto input_index{node.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto input_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)}; // optional
  const auto input_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_FORGET_WEIGHTS)};
  const auto input_to_cell_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_CELL_WEIGHTS)};
  const auto input_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto recurrent_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)}; // optional
  const auto recurrent_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS)};
  const auto recurrent_to_cell_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)};
  const auto recurrent_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto cell_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_INPUT_WEIGHTS)}; // optional
  const auto cell_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_FORGET_WEIGHTS)}; // optional
  const auto cell_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_OUTPUT_WEIGHTS)}; // optional
  const auto input_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_GATE_BIAS)};
  const auto forget_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::FORGET_GATE_BIAS)};
  const auto cell_gate_bias_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_BIAS)};
  const auto output_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_GATE_BIAS)};
  const auto projection_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_WEIGHTS)}; // optional
  const auto projection_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_BIAS)}; // optional
  const auto output_state_in_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_STATE_IN)};
  const auto cell_state_in_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_STATE_IN)};
  const auto time_major = node.param().time_major;

  // NOTE The input_to_input_weights and the recurrent_to_input_weights do not exist in CIFG.
  // has_input_to_input_weights && has_recurrent_to_input_weights: no CIFG
  // !(has_input_to_input_weights && has_recurrent_to_input_weights): CIFG
  // NOTE The cell_to_input_weights does not exist in non-peephole although regular LSTM(non-CIFG).
  bool has_input_to_input_weights = _ctx.exist(input_to_input_weights_index) &&
                                    (_ctx.at(input_to_input_weights_index).shape().dim(0) != 0 &&
                                     _ctx.at(input_to_input_weights_index).shape().dim(1) != 0);
  bool has_recurrent_to_input_weights =
    _ctx.exist(recurrent_to_input_weights_index) &&
    (_ctx.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
     _ctx.at(recurrent_to_input_weights_index).shape().dim(1) != 0);

  // NOTE The cell_to_forget_weights and the cell_to_output_weights exist in peephole.
  // But the cell_to_input_weights does not exist in regular CIFG although peephole.
  // has_cell_to_forget_weights && has_cell_to_output_weights: peephole
  // !(has_cell_to_forget_weights && has_cell_to_output_weights): no peephole
  bool has_cell_to_forget_weights = _ctx.exist(cell_to_forget_weights_index) &&
                                    _ctx.at(cell_to_forget_weights_index).shape().dim(0) != 0;
  bool has_cell_to_output_weights = _ctx.exist(cell_to_output_weights_index) &&
                                    _ctx.at(cell_to_output_weights_index).shape().dim(0) != 0;

  bool has_input_gate_bias =
    _ctx.exist(input_gate_bias_index) && _ctx.at(input_gate_bias_index).shape().dim(0);

  bool has_projection_weights = _ctx.exist(projection_weights_index) &&
                                (_ctx.at(projection_weights_index).shape().dim(0) != 0 &&
                                 _ctx.at(projection_weights_index).shape().dim(1) != 0);
  bool has_projection_bias =
    _ctx.exist(projection_bias_index) && _ctx.at(projection_bias_index).shape().dim(0);

  auto scratch_buffer_tensor = _ctx.exist(scratch_buffer_index)
                                 ? _tensor_reg->getPortableTensor(scratch_buffer_index)
                                 : nullptr; // optional
  auto output_state_out_tensor = _ctx.exist(output_state_out_index)
                                   ? _tensor_reg->getPortableTensor(output_state_out_index)
                                   : nullptr; // optional
  auto cell_state_out_tensor = _ctx.exist(cell_state_out_index)
                                 ? _tensor_reg->getPortableTensor(cell_state_out_index)
                                 : nullptr; // optional
  auto output_tensor = _tensor_reg->getPortableTensor(output_index);

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto input_to_input_weights_tensor =
    has_input_to_input_weights ? _tensor_reg->getPortableTensor(input_to_input_weights_index)
                               : nullptr; // optional
  auto input_to_forget_weights_tensor =
    _tensor_reg->getPortableTensor(input_to_forget_weights_index);
  auto input_to_cell_weights_tensor = _tensor_reg->getPortableTensor(input_to_cell_weights_index);
  auto input_to_output_weights_tensor =
    _tensor_reg->getPortableTensor(input_to_output_weights_index);
  auto recurrent_to_input_weights_tensor =
    has_recurrent_to_input_weights
      ? _tensor_reg->getPortableTensor(recurrent_to_input_weights_index)
      : nullptr; // optional
  auto recurrent_to_forget_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_forget_weights_index);
  auto recurrent_to_cell_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_cell_weights_index);
  auto recurrent_to_output_weights_tensor =
    _tensor_reg->getPortableTensor(recurrent_to_output_weights_index);

  auto cell_to_input_weights_tensor = _tensor_reg->getPortableTensor(cell_to_input_weights_index);
  auto cell_to_forget_weights_tensor =
    has_cell_to_forget_weights ? _tensor_reg->getPortableTensor(cell_to_forget_weights_index)
                               : nullptr; // optional
  auto cell_to_output_weights_tensor =
    has_cell_to_output_weights ? _tensor_reg->getPortableTensor(cell_to_output_weights_index)
                               : nullptr; // optional

  auto input_gate_bias_tensor =
    has_input_gate_bias ? _tensor_reg->getPortableTensor(input_gate_bias_index) : nullptr;
  auto forget_gate_bias_tensor = _tensor_reg->getPortableTensor(forget_gate_bias_index);
  auto cell_gate_bias_tensor = _tensor_reg->getPortableTensor(cell_gate_bias_index);
  auto output_gate_bias_tensor = _tensor_reg->getPortableTensor(output_gate_bias_index);
  auto output_state_in_tensor = _tensor_reg->getPortableTensor(output_state_in_index);
  auto cell_state_in_tensor = _tensor_reg->getPortableTensor(cell_state_in_index);

  auto projection_weights_tensor = has_projection_weights
                                     ? _tensor_reg->getPortableTensor(projection_weights_index)
                                     : nullptr; // optional
  auto projection_bias_tensor = has_projection_bias
                                  ? _tensor_reg->getPortableTensor(projection_bias_index)
                                  : nullptr; // optional

  IPortableTensor *input_layer_norm_weights_tensor = nullptr;
  IPortableTensor *forget_layer_norm_weights_tensor = nullptr;
  IPortableTensor *cell_layer_norm_weights_tensor = nullptr;
  IPortableTensor *output_layer_norm_weights_tensor = nullptr;
  if (node.getInputs().size() == 24)
  {
    const auto input_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_LAYER_NORMALIZATION_WEIGHTS)};
    const auto forget_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::FORGET_LAYER_NORMALIZATION_WEIGHTS)};
    const auto cell_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_LAYER_NORMALIZATION_WEIGHTS)};
    const auto output_layer_norm_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_LAYER_NORMALIZATION_WEIGHTS)};

    input_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(input_layer_norm_weights_index);
    forget_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(forget_layer_norm_weights_index);
    cell_layer_norm_weights_tensor = _tensor_reg->getPortableTensor(cell_layer_norm_weights_index);
    output_layer_norm_weights_tensor =
      _tensor_reg->getPortableTensor(output_layer_norm_weights_index);
  }

  auto fn = std::make_unique<ops::LSTMLayer>();

  fn->configure(
    input_tensor, input_to_input_weights_tensor, input_to_forget_weights_tensor,
    input_to_cell_weights_tensor, input_to_output_weights_tensor, recurrent_to_input_weights_tensor,
    recurrent_to_forget_weights_tensor, recurrent_to_cell_weights_tensor,
    recurrent_to_output_weights_tensor, cell_to_input_weights_tensor, cell_to_forget_weights_tensor,
    cell_to_output_weights_tensor, input_layer_norm_weights_tensor,
    forget_layer_norm_weights_tensor, cell_layer_norm_weights_tensor,
    output_layer_norm_weights_tensor,
    /*aux_input=*/nullptr,
    /*aux_input_to_input_weights=*/nullptr,
    /*aux_input_to_forget_weights=*/nullptr,
    /*aux_input_to_cell_weights=*/nullptr,
    /*aux_input_to_output_weights=*/nullptr, input_gate_bias_tensor, forget_gate_bias_tensor,
    cell_gate_bias_tensor, output_gate_bias_tensor, projection_weights_tensor,
    projection_bias_tensor, output_state_in_tensor, cell_state_in_tensor, node.param(),
    /*forward_sequence=*/true, time_major,
    /*output_offset=*/0, scratch_buffer_tensor, output_state_out_tensor, cell_state_out_tensor,
    output_tensor,
    !_ctx.at(output_state_in_index).info().isVariable() /* means empty buffer on frontend now */,
    !_ctx.at(cell_state_in_index).info().isVariable());

  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::LSTM &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

namespace
{
template <typename T>
T *getOptionalOutputBuffer(onert::backend::IPortableTensor *tensor, std::vector<uint8_t> *temp_vec,
                           size_t total_size)
{
  if (tensor == nullptr)
  {
    temp_vec->reserve(total_size);
    return reinterpret_cast<T *>(temp_vec->data());
  }
  else
  {
    assert(tensor->total_size() == total_size);
    return getBuffer<T>(tensor);
  }
}

inline void initializeStateBuffer(const onert::backend::IPortableTensor *tensor_in, void *buffer,
                                  bool needs_memcpy)
{
  assert(tensor_in != nullptr);
  assert(buffer != nullptr);
  if (needs_memcpy)
    memcpy(buffer, tensor_in->buffer(), tensor_in->total_size());
  else
    memset(buffer, 0, tensor_in->total_size());
}
} // namespace

void LSTMLayer::LSTMFloat()
{
  auto in_shape = _input->getShape();
  assert(in_shape.rank() >= 2 && in_shape.rank() <= 3);
  int max_time, n_batch;
  if (in_shape.rank() == 3)
  {
    max_time = (_time_major) ? in_shape.dim(0) : in_shape.dim(1);
    n_batch = (_time_major) ? in_shape.dim(1) : in_shape.dim(0);
  }
  else
  {
    max_time = 1;
    n_batch = in_shape.dim(0);
  }
  const int n_input = in_shape.dim(_input->getShape().rank() - 1);
  const int aux_input_size = 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = _input_to_output_weights->getShape().dim(0);
  const int n_output = _recurrent_to_output_weights->getShape().dim(1);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (_input_to_input_weights == nullptr);

  // Optional outputs
  float *output_state_buf = getOptionalOutputBuffer<float>(_output_state, &_output_state_vec,
                                                           _output_state_in->total_size());
  float *cell_state_buf =
    getOptionalOutputBuffer<float>(_cell_state, &_cell_state_vec, _cell_state_in->total_size());

  initializeStateBuffer(_output_state_in, output_state_buf, _has_output_state_data);
  initializeStateBuffer(_cell_state_in, cell_state_buf, _has_cell_state_data);

  // Index the scratch buffers pointers to the global scratch buffer.
  float *scratch_buffer_buf = getOptionalOutputBuffer<float>(
    _scratch_buffer, &_scratch_vec, n_batch * n_cell * (use_cifg ? 3 : 4) * sizeof(float));
  float *input_gate_scratch = nullptr;
  float *cell_gate_scratch = nullptr;
  float *forget_gate_scratch = nullptr;
  float *output_gate_scratch = nullptr;
  if (use_cifg)
  {
    cell_gate_scratch = scratch_buffer_buf;
    forget_gate_scratch = scratch_buffer_buf + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_buf + 2 * n_cell * n_batch;
  }
  else
  {
    input_gate_scratch = scratch_buffer_buf;
    cell_gate_scratch = scratch_buffer_buf + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_buf + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_buf + 3 * n_cell * n_batch;
  }

  auto optional_tensor_ptr = [](const IPortableTensor *tensor) {
    // If tensor is not given or the tensor size is 0, consider it was not given
    return (tensor && tensor->total_size() > 0) ? getBuffer<float>(tensor) : nullptr;
  };
  // Optional inputs
  const float *input_to_input_weights_ptr = optional_tensor_ptr(_input_to_input_weights);
  const float *recurrent_to_input_weights_ptr = optional_tensor_ptr(_recurrent_to_input_weights);
  const float *cell_to_input_weights_ptr = optional_tensor_ptr(_cell_to_input_weights);
  const float *cell_to_forget_weights_ptr = optional_tensor_ptr(_cell_to_forget_weights);
  const float *cell_to_output_weights_ptr = optional_tensor_ptr(_cell_to_output_weights);
  const float *input_gate_bias_ptr = optional_tensor_ptr(_input_gate_bias);
  const float *projection_weights_ptr = optional_tensor_ptr(_projection_weights);
  const float *projection_bias_ptr = optional_tensor_ptr(_projection_bias);
  const float *input_layer_norm_coefficients_ptr =
    optional_tensor_ptr(_input_layer_norm_coefficients);
  const float *forget_layer_norm_coefficients_ptr =
    optional_tensor_ptr(_forget_layer_norm_coefficients);
  const float *cell_layer_norm_coefficients_ptr =
    optional_tensor_ptr(_cell_layer_norm_coefficients);
  const float *output_layer_norm_coefficients_ptr =
    optional_tensor_ptr(_output_layer_norm_coefficients);

  // Copy out the LSTM specific params so they can be passed in the function.
  nnfw::cker::LSTMParams lstm_params;
  lstm_params.activation = convertActivationType(_params.activation);
  lstm_params.cell_clip = _params.cell_threshold;
  lstm_params.proj_clip = _params.projection_threshold;

  auto out_shape = _output->getShape();
  const int output_batch_leading_dim = out_shape.dim(out_shape.rank() - 1);
  if (_time_major)
  {
    // Loop through the sequence.
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++)
    {
      // If this is the forward_sequence, step forward, otherwise step
      // backwards.
      const int t_rel = _forward_sequence ? t : max_time - t - 1;
      const float *input_ptr = getBuffer<float>(_input) + t_rel * input_step;
      const float *aux_input_ptr = nullptr;
      if (_aux_input)
      {
        aux_input_ptr = getBuffer<float>(_aux_input) + t_rel * input_step;
      }
      float *output_ptr = getBuffer<float>(_output) + t_rel * output_step + _output_offset;

      LstmStepFloat(
        input_ptr, input_to_input_weights_ptr, getBuffer<float>(_input_to_forget_weights),
        getBuffer<float>(_input_to_cell_weights), getBuffer<float>(_input_to_output_weights),
        aux_input_ptr,
        /*aux_input_to_input_weights=*/nullptr,
        /*aux_input_to_forget_weights=*/nullptr,
        /*aux_input_to_cell_weights=*/nullptr,
        /*aux_input_to_output_weights=*/nullptr, recurrent_to_input_weights_ptr,
        getBuffer<float>(_recurrent_to_forget_weights),
        getBuffer<float>(_recurrent_to_cell_weights),
        getBuffer<float>(_recurrent_to_output_weights), cell_to_input_weights_ptr,
        cell_to_forget_weights_ptr, cell_to_output_weights_ptr, input_layer_norm_coefficients_ptr,
        forget_layer_norm_coefficients_ptr, cell_layer_norm_coefficients_ptr,
        output_layer_norm_coefficients_ptr, input_gate_bias_ptr,
        getBuffer<float>(_forget_gate_bias), getBuffer<float>(_cell_gate_bias),
        getBuffer<float>(_output_gate_bias), projection_weights_ptr, projection_bias_ptr,
        &lstm_params, n_batch, n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
        output_state_buf, cell_state_buf, input_gate_scratch, forget_gate_scratch,
        cell_gate_scratch, output_gate_scratch, output_ptr);
    }
  }
  else
  {
    for (int b = 0; b < n_batch; b++)
    {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++)
      {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = _forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float *input_ptr = getBuffer<float>(_input) + time_offset * input_step;
        const float *aux_input_ptr = nullptr;
        if (_aux_input)
        {
          aux_input_ptr = getBuffer<float>(_aux_input) + time_offset * input_step;
        }
        float *output_ptr = getBuffer<float>(_output) + time_offset * output_step + _output_offset;

        // Offset the {output,cell}_state pointers to the right batch.
        float *output_state_ptr = output_state_buf + b * output_batch_leading_dim;
        float *cell_state_ptr = cell_state_buf + b * n_cell;
        // Offset the scratch pointers to the right batch.
        float *input_gate_scratch_ptr =
          input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float *forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float *cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        float *output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        LstmStepFloat(
          input_ptr, input_to_input_weights_ptr, getBuffer<float>(_input_to_forget_weights),
          getBuffer<float>(_input_to_cell_weights), getBuffer<float>(_input_to_output_weights),
          aux_input_ptr,
          /*aux_input_to_input_weights=*/nullptr,
          /*aux_input_to_forget_weights=*/nullptr,
          /*aux_input_to_cell_weights=*/nullptr,
          /*aux_input_to_output_weights=*/nullptr, recurrent_to_input_weights_ptr,
          getBuffer<float>(_recurrent_to_forget_weights),
          getBuffer<float>(_recurrent_to_cell_weights),
          getBuffer<float>(_recurrent_to_output_weights), cell_to_input_weights_ptr,
          cell_to_forget_weights_ptr, cell_to_output_weights_ptr, input_layer_norm_coefficients_ptr,
          forget_layer_norm_coefficients_ptr, cell_layer_norm_coefficients_ptr,
          output_layer_norm_coefficients_ptr, input_gate_bias_ptr,
          getBuffer<float>(_forget_gate_bias), getBuffer<float>(_cell_gate_bias),
          getBuffer<float>(_output_gate_bias), projection_weights_ptr, projection_bias_ptr,
          &lstm_params, /*n_batch=*/1, n_cell, n_input, aux_input_size, n_output,
          output_batch_leading_dim, output_state_ptr, cell_state_ptr, input_gate_scratch_ptr,
          forget_gate_scratch_ptr, cell_gate_scratch_ptr, output_gate_scratch_ptr, output_ptr);
      }
    }
  }
}

void LSTMLayer::configure(
  const IPortableTensor *input, const IPortableTensor *input_to_input_weights,
  const IPortableTensor *input_to_forget_weights, const IPortableTensor *input_to_cell_weights,
  const IPortableTensor *input_to_output_weights, const IPortableTensor *recurrent_to_input_weights,
  const IPortableTensor *recurrent_to_forget_weights,
  const IPortableTensor *recurrent_to_cell_weights,
  const IPortableTensor *recurrent_to_output_weights, const IPortableTensor *cell_to_input_weights,
  const IPortableTensor *cell_to_forget_weights, const IPortableTensor *cell_to_output_weights,
  const IPortableTensor *input_layer_norm_weights, const IPortableTensor *forget_layer_norm_weights,
  const IPortableTensor *cell_layer_norm_weights, const IPortableTensor *output_layer_norm_weights,
  const IPortableTensor *aux_input, const IPortableTensor *aux_input_to_input_weights,
  const IPortableTensor *aux_input_to_forget_weights,
  const IPortableTensor *aux_input_to_cell_weights,
  const IPortableTensor *aux_input_to_output_weights, const IPortableTensor *input_gate_bias,
  const IPortableTensor *forget_gate_bias, const IPortableTensor *cell_gate_bias,
  const IPortableTensor *output_gate_bias, const IPortableTensor *projection_weights,
  const IPortableTensor *projection_bias, const IPortableTensor *output_state_in,
  const IPortableTensor *cell_state_in, const ir::operation::LSTM::Param &params,
  bool forward_sequence, bool time_major, int output_offset, IPortableTensor *scratch_buffer,
  IPortableTensor *output_state, IPortableTensor *cell_state, IPortableTensor *output,
  bool has_output_state_data, bool has_cell_state_data)
{
  _input = input;
  _input_to_input_weights = input_to_input_weights;
  _input_to_forget_weights = input_to_forget_weights;
  _input_to_cell_weights = input_to_cell_weights;
  _input_to_output_weights = input_to_output_weights;
  _recurrent_to_input_weights = recurrent_to_input_weights;
  _recurrent_to_forget_weights = recurrent_to_forget_weights;
  _recurrent_to_cell_weights = recurrent_to_cell_weights;
  _recurrent_to_output_weights = recurrent_to_output_weights;
  _cell_to_input_weights = cell_to_input_weights;
  _cell_to_forget_weights = cell_to_forget_weights;
  _cell_to_output_weights = cell_to_output_weights;
  _input_layer_norm_coefficients = input_layer_norm_weights;
  _forget_layer_norm_coefficients = forget_layer_norm_weights;
  _cell_layer_norm_coefficients = cell_layer_norm_weights;
  _output_layer_norm_coefficients = output_layer_norm_weights;
  _aux_input = aux_input, _aux_input_to_input_weights = aux_input_to_input_weights,
  _aux_input_to_forget_weights = aux_input_to_forget_weights,
  _aux_input_to_cell_weights = aux_input_to_cell_weights,
  _aux_input_to_output_weights = aux_input_to_output_weights, _input_gate_bias = input_gate_bias;
  _forget_gate_bias = forget_gate_bias;
  _cell_gate_bias = cell_gate_bias;
  _output_gate_bias = output_gate_bias;
  _projection_weights = projection_weights;
  _projection_bias = projection_bias;
  _output_state_in = output_state_in;
  _cell_state_in = cell_state_in;
  _params = params;
  _forward_sequence = forward_sequence;
  _time_major = time_major;
  _output_offset = output_offset;
  _scratch_buffer = scratch_buffer;
  _output_state = output_state;
  _cell_state = cell_state;
  _output = output;
  _has_output_state_data = has_output_state_data;
  _has_cell_state_data = has_cell_state_data;
}

void LSTMLayer::run()
{

  if (_input->data_type() == OperandType::FLOAT32)
  {
    LSTMFloat();
  }
  else
  {
    throw std::runtime_error{"LSTMLayer: unsupported data type"};
  }
}

} // namespace onert::backend::cpu::ops
