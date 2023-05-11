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

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
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

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
