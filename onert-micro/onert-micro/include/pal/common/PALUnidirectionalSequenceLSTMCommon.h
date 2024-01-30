/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H

#include "core/OMUtils.h"
#include "PALUnidirectionalSequenceLSTMKernel.h"
#include "core/OMRuntimeContext.h"
#include "core/OMRuntimeStorage.h"
#include "core/OMKernelData.h"
#include "PALTanh.h"
#include "PALLogistic.h"
#include "PALFullyConnected.h"
#include "PALMul.h"
#include "PALUtils.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace onert_micro
{
namespace execute
{
namespace pal
{
namespace lstm_internal
{
namespace
{
// Possible fused activation functions.
typedef enum
{
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1, // min(max(-1, x), 1)
  kTfLiteActRelu6,     // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} FusedActivation;

} // namespace

#ifndef DIS_QUANT

template <typename InputType, typename OutputType>
void mulElementwise(int size, const lstm::ArithmeticParams *params, const InputType *input1_data,
                    const InputType *input2_data, OutputType *output_data)
{
  for (int i = 0; i < size; ++i)
  {
    const int32_t input1_val = params->input1_offset + input1_data[i];
    const int32_t input2_val = params->input2_offset + input2_data[i];
    const int32_t unclamped_result =
      params->output_offset + multiplyByQuantizedMultiplier(input1_val * input2_val,
                                                            params->output_multiplier,
                                                            params->output_shift);
    const int32_t clamped_output =
      std::min(params->quantized_activation_max,
               std::max(params->quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<OutputType>(clamped_output);
  }
}

// Input and output have the same shape in LSTM
void mul(const core::OMRuntimeShape &shape, const lstm::ArithmeticParams *params,
         const int16_t *input1_data, const int16_t *input2_data, int8_t *output_data)
{
  return mulElementwise<int16_t, int8_t>(shape.flatSize(), params, input1_data, input2_data,
                                         output_data);
}

// Input and output have the same shape in LSTM
void mul(const core::OMRuntimeShape &shape, const lstm::ArithmeticParams *params,
         const int16_t *input1_data, const int16_t *input2_data, int16_t *output_data)
{
  return mulElementwise(shape.flatSize(), params, input1_data, input2_data, output_data);
}

void addElementWise(const int16_t *input_1, const int16_t *input_2, int n_batch, int n_input,
                    int16_t *output)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32_t sum_clamped =
        std::min(static_cast<int32_t>(std::numeric_limits<int16_t>::max()),
                 std::max(static_cast<int32_t>(std::numeric_limits<int16_t>::min()), sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

void tanh(int32_t cell_state_scale_power, const core::OMRuntimeShape &input_data_shape,
          int16_t *input_data, const core::OMRuntimeShape &output_data_shape, int16_t *output_data)
{
  int32_t tanh_input_left_shift = (15 + cell_state_scale_power) - 3;
  int32_t input_multiplier = 0;
  if (tanh_input_left_shift < 0) /* handling negative shift value */
  {
    tanh_input_left_shift = -tanh_input_left_shift;
    input_multiplier = 3;
  }
  const int flat_size = input_data_shape.flatSize();
  pal::Tanh(input_multiplier, tanh_input_left_shift, flat_size, input_data, output_data);
}

void sigmoid(const core::OMRuntimeShape &data_shape, int16_t *data)
{
  pal::Logistic(0, 0, data_shape.flatSize(), data, data);
}

void clipping(const int v_size, const lstm::CellStateInfo *cell_state_info, int16_t *vector)
{
  for (int i = 0; i < v_size; i++)
  {
    vector[i] = std::max(std::min(cell_state_info->quantized_cell_clip, vector[i]),
                         static_cast<int16_t>(-cell_state_info->quantized_cell_clip));
  }
}
#endif // DIS_QUANT

#ifndef DIS_FLOAT
// Input and output have the same shape in LSTM
void mul(const core::OMRuntimeShape &shape, const lstm::ArithmeticParams *params,
         const float *input1_data, const float *input2_data, float *output_data)
{
  const int flat_size = shape.flatSize();
  core::BinaryArithmeticBroadcastParams op_params{};
  op_params.float_activation_max = params->float_activation_max;
  op_params.float_activation_min = params->float_activation_min;

  pal::Mul(op_params, flat_size, input1_data, input2_data, output_data);
}

void addElementWise(const float *input_1, const float *input_2, int n_batch, int n_input,
                    float *output)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      output[index] = input_1[index] + input_2[index];
    }
  }
}

void tanh(int32_t, const core::OMRuntimeShape &input_data_shape, float *input_data,
          const core::OMRuntimeShape &output_data_shape, float *output_data)
{
  const int flat_size = input_data_shape.flatSize();
  pal::Tanh(flat_size, input_data, output_data);
}

void sigmoid(const core::OMRuntimeShape &data_shape, float *data)
{
  const int flat_size = data_shape.flatSize();
  pal::Logistic(flat_size, data, data);
}

void clipping(const int v_size, const lstm::CellStateInfo *cell_state_info, float *vector)
{
  for (int i = 0; i < v_size; i++)
  {
    vector[i] =
      std::max(std::min(cell_state_info->cell_clip, vector[i]), -cell_state_info->cell_clip);
  }
}
#endif // DIS_FLOAT

// Size information about the LSTM kernel, which is deduced from tensors stored
// in the flat buffer file.
struct LstmSizeInfo
{
  bool time_major;
  int32_t batch_size;
  int32_t time_steps;
  int32_t input_dimension;
  int32_t state_dimension;
};

class LstmStepManager
{
public:
  LstmStepManager() = delete;
  // Does not take any ownership, and all pointers must refer to valid objects
  // that outlive the one constructed.
  explicit LstmStepManager(const LstmSizeInfo &size_info) : size_info_(size_info) {}

  void updateTime()
  {
    current_time_ += 1;
    // default as one batch per inference
    int input_step = size_info_.input_dimension;
    int output_step = size_info_.state_dimension;
    // time major: batch inference
    if (size_info_.time_major)
    {
      input_step = input_step * size_info_.batch_size;
      output_step = output_step * size_info_.batch_size;
    }

    input_offset_ += input_step;
    output_offset_ += output_step;
  }

  void updateBatch()
  {
    current_batch_ += 1;
    // batch inference for time major: no action needed
    if (size_info_.time_major)
    {
      return;
    }
    // otherwise: singe batch inference, go to the next batch
    hidden_state_offset_ += size_info_.state_dimension;
    cell_state_offset_ += size_info_.state_dimension;
  }

  void resetTime() { current_time_ = 0; }

  core::OMRuntimeShape inputShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.input_dimension};
    const int32_t *dims_data = reinterpret_cast<const int32_t *>(dims);
    return core::OMRuntimeShape(2, dims_data);
  }

  core::OMRuntimeShape stateShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.state_dimension};
    const int32_t *dims_data = reinterpret_cast<const int32_t *>(dims);
    return {2, dims_data};
  }

  int inputOffset() const { return input_offset_; }

  int outputOffset() const { return output_offset_; }

  int hiddenStateOffset() const { return hidden_state_offset_; }

  int cellStateOffset() const { return cell_state_offset_; }

private:
  int32_t current_time_ = 0;
  int32_t current_batch_ = 0;
  int32_t input_offset_ = 0;
  int32_t output_offset_ = 0;
  int32_t hidden_state_offset_ = 0;
  int32_t cell_state_offset_ = 0;

  const LstmSizeInfo &size_info_;
};

// Calculates a single LSTM gate.
// Implements the following formula:
//   gate = activate(FC(input) + FC(recurrent))
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
OMStatus
calculateLstmGate(const LstmStepManager *step_info, const lstm::GateParameters *gate_params,
                  // Input FC
                  ActivationType *input_data, const core::OMRuntimeShape &input_weight_shape,
                  uint8_t *input_weight_data, uint8_t *input_bias_data,
                  // Recurrent FC
                  ActivationType *recurrent_data,
                  const core::OMRuntimeShape &recurrent_weight_shape,
                  uint8_t *recurrent_weight_data, uint8_t *recurrent_bias_data,
                  // Output
                  CellType *gate_output,
                  // Scratch arrays
                  CellType *fc_output_buffer, const FusedActivation activation)
{
  OMStatus status = Ok;

  // Input FC
  const auto gate_output_shape = step_info->stateShape();
  {
    core::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params->input_fc_params.input_offset;
    op_params.weights_offset = gate_params->input_fc_params.weights_offset;
    op_params.output_offset = gate_params->input_fc_params.output_offset;
    op_params.output_multiplier = gate_params->input_fc_params.output_multiplier;
    op_params.output_shift = gate_params->input_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params->input_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params->input_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params->input_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params->input_fc_params.float_activation_min;

    const ActivationType *input_data_offseted =
      const_cast<const ActivationType *>(input_data + step_info->inputOffset());
    status = pal::FullyConnected(op_params, input_data_offseted, input_weight_shape,
                                 core::utils::castInputData<WeightType>(input_weight_data),
                                 core::utils::castInputData<BiasType>(input_bias_data),
                                 gate_output_shape, gate_output);
  }

  if (status != Ok)
    return status;

  // Recurrent FC
  {
    core::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params->recurrent_fc_params.input_offset;
    op_params.weights_offset = gate_params->recurrent_fc_params.weights_offset;
    op_params.output_offset = gate_params->recurrent_fc_params.output_offset;
    op_params.output_multiplier = gate_params->recurrent_fc_params.output_multiplier;
    op_params.output_shift = gate_params->recurrent_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params->recurrent_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params->recurrent_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params->recurrent_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params->recurrent_fc_params.float_activation_min;

    status = pal::FullyConnected(op_params, recurrent_data + step_info->hiddenStateOffset(),
                                 recurrent_weight_shape,
                                 core::utils::castInputData<WeightType>(recurrent_weight_data),
                                 core::utils::castInputData<BiasType>(recurrent_bias_data),
                                 gate_output_shape, fc_output_buffer);

    addElementWise(gate_output, fc_output_buffer, /*n_batch=*/gate_output_shape.dimsData()[0],
                   /*n_state=*/gate_output_shape.dimsData()[1], gate_output);

    switch (activation)
    {
      case FusedActivation::kTfLiteActSigmoid:
        sigmoid(gate_output_shape, gate_output);
        break;
      case FusedActivation::kTfLiteActTanh:
      {
        // Set the scale power to -12 to avoid shift
        tanh(/*cell_state_scale_power=*/-12, gate_output_shape, gate_output, gate_output_shape,
             gate_output);
      }
      break;
      default:
        // Only Sigmoid or Tanh is used.
        assert(false && "Only Sigmoid or Tanh is used");
    }
  }
  return status;
}

// Update the hidden state of the LSTM kernel using the following formula:
// updated_hidden_state = Tanh(updated_cell_state) * output_gate_output, * means
// element wise multiplication
template <typename CellType, typename ActivationType>
void updateLstmHidden(const LstmStepManager *step_info, CellType *cell_state_data_base,
                      ActivationType *hidden_state_data, const CellType *output_gate_output,
                      const lstm::ArithmeticParams *mul_params, int32_t cell_state_scale_power,
                      CellType *buffer)
{
  auto cell_state_shape = step_info->stateShape();
  CellType *cell_state_data = cell_state_data_base + step_info->cellStateOffset();
  // Tanh(cell_state)
  tanh(cell_state_scale_power, cell_state_shape, cell_state_data, cell_state_shape, buffer);
  // Update the hidden state
  mul(cell_state_shape, mul_params, buffer, output_gate_output,
      hidden_state_data + step_info->hiddenStateOffset());
}

// Update the cell state using the output from the forget gate, input gate, and
// cell gate Formula: updated_cell_state = forget_gate_output*cell_state +
// input_gate_output * cell_gate_output, where * denotes element wise
// multiplication
template <typename CellType>
void updateLstmCell(const LstmStepManager *step_info, CellType *cell_state_data,
                    // Gate outputs
                    CellType *forget_gate_output, const CellType *input_gate_output,
                    const CellType *cell_gate_output,
                    // Mul parameters
                    const lstm::ArithmeticParams &forget_cell_mul_params,
                    const lstm::ArithmeticParams &input_mul_params,
                    const lstm::CellStateInfo *cell_state_info, CellType *buffer)
{
  auto cell_state_shape = step_info->stateShape();
  // Forget Gate x Cell State
  mul(cell_state_shape, &forget_cell_mul_params, forget_gate_output,
      cell_state_data + step_info->cellStateOffset(),
      cell_state_data + step_info->cellStateOffset());
  // Input Gate x Cell Gate
  mul(cell_state_shape, &input_mul_params, input_gate_output, cell_gate_output, buffer);

  // Update the cell state
  addElementWise(cell_state_data + step_info->cellStateOffset(), buffer,
                 /*n_batch=*/cell_state_shape.dimsData()[0],
                 /*n_state=*/cell_state_shape.dimsData()[1],
                 cell_state_data + step_info->cellStateOffset());

  if (cell_state_info->cell_clip > 0)
  {
    clipping(cell_state_shape.flatSize(), cell_state_info,
             cell_state_data + step_info->cellStateOffset());
  }
}

template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
void lstmStep(lstm::LSTMStruct *lstm_struct, lstm::LSTMParameters *lstm_params,
              LstmStepManager *step_info, lstm::CellStateInfo *cell_state_info,
              ActivationType *output_state_data, CellType *cell_state_data, CellType *scratch0,
              CellType *scratch1, CellType *scratch2, CellType *scratch3,
              core::OMRuntimeStorage &runtime_storage, core::OMRuntimeContext &runtime_context)
{
  /*Step1: Calculate gate outputs to prepare cell state update*/
  CellType *gate_internal_buffer = scratch3;
  CellType *forget_gate_output = scratch0;

  auto *input_data = core::utils::castOutputData<ActivationType>(lstm_struct->inputs_data[0]);
  // core::utils::castInputData<ActivationType>(runtime_storage.getDataByTensorIndex(&input_data,
  // lstm_struct->inputs_index[0]));

  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->forget_gate_parameters,
    // Input FC
    input_data, core::OMRuntimeShape(lstm_struct->input_to_forget_weights()),
    lstm_struct->inputs_data[2], lstm_struct->inputs_data[13], // lstm_struct->forget_gate_bias(),
    // Recurrent FC
    output_state_data, core::OMRuntimeShape(lstm_struct->recurrent_to_forget_weights()),
    lstm_struct->inputs_data[6], nullptr,
    // Output
    forget_gate_output, gate_internal_buffer, FusedActivation::kTfLiteActSigmoid);

  // Input Gate calculation;
  CellType *input_gate_output = scratch1;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->input_gate_parameters,
    // Input FC
    input_data, core::OMRuntimeShape(lstm_struct->input_to_input_weights()),
    lstm_struct->inputs_data[1], lstm_struct->inputs_data[12], // lstm_struct->input_gate_bias(),
    // Recurrent FC
    output_state_data, core::OMRuntimeShape(lstm_struct->recurrent_to_input_weights()),
    lstm_struct->inputs_data[5],
    /*recurrent_bias*/ nullptr,
    // Output
    input_gate_output,
    // Scratch arrays
    gate_internal_buffer, FusedActivation::kTfLiteActSigmoid);

  // Cell Gate calculation
  CellType *cell_gate_output = scratch2;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->cell_gate_parameters,
    // Input FC
    input_data, core::OMRuntimeShape(lstm_struct->input_to_cell_weights()),
    lstm_struct->inputs_data[3], lstm_struct->inputs_data[14], // lstm_struct->cell_gate_bias(),
    // Recurrent FC
    output_state_data, core::OMRuntimeShape(lstm_struct->recurrent_to_cell_weights()),
    lstm_struct->inputs_data[7],
    /*recurrent_bias*/ nullptr,
    // Output
    cell_gate_output,
    // Scratch arrays
    gate_internal_buffer, FusedActivation::kTfLiteActTanh);

  /*Step2: update the cell state */
  {
    // const InterGateParameters& inter_gate_params = op_data.inter_gate_parameters;
    CellType *updated_input_buffer = scratch1; // reuse buffer

    updateLstmCell<CellType>(
      step_info, cell_state_data, forget_gate_output, input_gate_output, cell_gate_output,
      lstm_params->inter_gate_parameters.forget_cell_mul_params,
      lstm_params->inter_gate_parameters.input_mul_params, cell_state_info, updated_input_buffer);
  }

  {
    /*Step3: update the hidden state */
    CellType *output_gate_output = scratch1; // reuse buffer
    calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
      step_info, &lstm_params->output_gate_parameters,
      // Input FC
      input_data, core::OMRuntimeShape(lstm_struct->input_to_output_weights()),
      lstm_struct->inputs_data[4], lstm_struct->inputs_data[15], // lstm_struct->output_gate_bias(),
      // Recurrent FC
      output_state_data, core::OMRuntimeShape(lstm_struct->recurrent_to_output_weights()),
      lstm_struct->inputs_data[8], nullptr,
      // Output
      output_gate_output,
      // Scratch arrays
      gate_internal_buffer, FusedActivation::kTfLiteActSigmoid);
    CellType *tanh_activated_cell_buffer = scratch0; // reuse buffer
    updateLstmHidden<CellType, ActivationType>(
      step_info, cell_state_data, output_state_data, output_gate_output,
      &lstm_params->inter_gate_parameters.output_mul_params,
      cell_state_info->cell_state_scale_power, tanh_activated_cell_buffer);

    ActivationType *output_ptr = core::utils::castOutputData<ActivationType>(
      lstm_struct->output_data); // luci_interpreter::kernels::getTensorData<ActivationType>(
    std::memcpy(output_ptr + step_info->outputOffset(),
                output_state_data + step_info->hiddenStateOffset(),
                step_info->stateShape().flatSize() * sizeof(ActivationType));
  }
}

} // namespace lstm_internal

// Evaluate the LSTM kernel with (potential) multi-steps and multi-batch input
template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
OMStatus evalLSTM(lstm::LSTMStruct *lstm_struct, lstm::LSTMParameters *lstm_params,
                  lstm::CellStateInfo *cell_state_info, ActivationType *output_state_data,
                  CellType *cell_state_data, CellType *scratch0, CellType *scratch1,
                  CellType *scratch2, CellType *scratch3, core::OMRuntimeStorage &runtime_storage,
                  core::OMRuntimeContext &runtime_context)
{
  lstm_internal::LstmSizeInfo size_info{};

  size_info.time_major = lstm_struct->options->time_major();
  size_info.batch_size = size_info.time_major
                           ? lstm_struct->input()->shape()->operator[](
                               1) // luci_interpreter::Tensor::dim(lstm_struct->input(), 1)
                           : lstm_struct->input()->shape()->operator[](
                               0); // luci_interpreter::Tensor::dim(lstm_struct->input(), 0);
  size_info.time_steps = size_info.time_major
                           ? lstm_struct->input()->shape()->operator[](
                               0) // luci_interpreter::Tensor::dim(lstm_struct->input(), 0)
                           : lstm_struct->input()->shape()->operator[](
                               1); // luci_interpreter::Tensor::dim(lstm_struct->input(), 1);
  size_info.input_dimension = lstm_struct->input()->shape()->operator[](
    2); // luci_interpreter::Tensor::dim(lstm_struct->input(), 2);
  size_info.state_dimension = lstm_struct->output_state()->shape()->operator[](
    1); // luci_interpreter::Tensor::dim(lstm_struct->output_state(), 1);

  lstm_internal::LstmStepManager step_info(size_info);

  // time is the first dimention, enable batch computation
  if (size_info.time_major)
  {
    for (int t = 0; t < size_info.time_steps; t++)
    {
      lstm_internal::lstmStep<ActivationType, WeightType, CellType, BiasType>(
        lstm_struct, lstm_params, &step_info, cell_state_info, output_state_data, cell_state_data,
        scratch0, scratch1, scratch2, scratch3, runtime_storage, runtime_context);
      // prepare for the next time step
      step_info.updateTime();
    }
  }
  else
  {
    // batch first, unable to size the input data. single batch inference
    for (int b = 0; b < size_info.batch_size; b++)
    {
      for (int t = 0; t < size_info.time_steps; t++)
      {
        lstm_internal::lstmStep<ActivationType, WeightType, CellType, BiasType>(
          lstm_struct, lstm_params, &step_info, cell_state_info, output_state_data, cell_state_data,
          scratch0, scratch1, scratch2, scratch3, runtime_storage, runtime_context);
        // prepare for the next time step
        step_info.updateTime();
      }
      // prepare for the next batch
      step_info.updateBatch();
      step_info.resetTime();
    }
  }
  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H
