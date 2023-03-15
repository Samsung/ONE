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

#ifndef LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "kernels/UnidirectionalSequenceLSTM.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"

namespace luci_interpreter_pal
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
} TfLiteFusedActivation;

} // namespace

template <typename T>
inline T activationFunctionWithMinMax(T x, T output_activation_min, T output_activation_max)
{
  using std::max;
  using std::min;
  return min(max(x, output_activation_min), output_activation_max);
}

template <typename T>
inline void mul(const luci_interpreter::lstm::ArithmeticParams *params,
                const tflite::RuntimeShape &input1_shape, const T *input1_data,
                const tflite::RuntimeShape &input2_shape, const T *input2_data,
                const tflite::RuntimeShape &output_shape, T *output_data)
{
  T output_activation_min = params->quantized_activation_min;
  T output_activation_max = params->quantized_activation_max;

  const int flat_size = input1_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = activationFunctionWithMinMax(input1_data[i] * input2_data[i],
                                                  output_activation_min, output_activation_max);
  }
}

#ifndef DIS_QUANT
inline int32_t multiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
    SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
void fullyConnectedInteger(const tflite::FullyConnectedParams &params,
                           const tflite::RuntimeShape &input_shape, const InputType *input_data,
                           const tflite::RuntimeShape &filter_shape, const WeightType *filter_data,
                           const tflite::RuntimeShape &bias_shape, const BiasType *bias_data,
                           const tflite::RuntimeShape &output_shape, OutputType *output_data)
{
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b)
  {
    for (int out_c = 0; out_c < output_depth; ++out_c)
    {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d)
      {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data)
      {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = multiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<OutputType>(acc_scaled);
    }
  }
}

void fullyConnected(const tflite::FullyConnectedParams &params,
                    const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                    const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                    const tflite::RuntimeShape &bias_shape, const int32_t *bias_data,
                    const tflite::RuntimeShape &output_shape, int16_t *output_data)
{
  return fullyConnectedInteger(params, input_shape, input_data, filter_shape, filter_data,
                               bias_shape, bias_data, output_shape, output_data);
}

void fullyConnected(const tflite::FullyConnectedParams &params,
                    const tflite::RuntimeShape &input_shape, const int16_t *input_data,
                    const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                    const tflite::RuntimeShape &bias_shape, const int64_t *bias_data,
                    const tflite::RuntimeShape &output_shape, int16_t *output_data)
{
  return fullyConnectedInteger(params, input_shape, input_data, filter_shape, filter_data,
                               bias_shape, bias_data, output_shape, output_data);
}

template <typename InputType, typename OutputType>
void mulElementwise(int size, const luci_interpreter::lstm::ArithmeticParams *params,
                    const InputType *input1_data, const InputType *input2_data,
                    OutputType *output_data)
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
void mul(const tflite::RuntimeShape &shape, const luci_interpreter::lstm::ArithmeticParams *params,
         const int16_t *input1_data, const int16_t *input2_data, int8_t *output_data)
{
  return mulElementwise<int16_t, int8_t>(shape.FlatSize(), params, input1_data, input2_data,
                                         output_data);
}

// Input and output have the same shape in LSTM
void mul(const tflite::RuntimeShape &shape, const luci_interpreter::lstm::ArithmeticParams *params,
         const int16_t *input1_data, const int16_t *input2_data, int16_t *output_data)
{
  return mulElementwise(shape.FlatSize(), params, input1_data, input2_data, output_data);
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

void tanh(int32_t cell_state_scale_power, const tflite::RuntimeShape &input_data_shape,
          int16_t *input_data, const tflite::RuntimeShape &output_data_shape, int16_t *output_data)
{
  int32_t tanh_input_left_shift = (15 + cell_state_scale_power) - 3;
  int32_t input_multiplier = 0;
  if (tanh_input_left_shift < 0) /* handling negative shift value */
  {
    tanh_input_left_shift = -tanh_input_left_shift;
    input_multiplier = 3;
  }
  tflite::reference_integer_ops::Tanh(input_multiplier, tanh_input_left_shift, input_data_shape,
                                      input_data, output_data_shape, output_data);
}

void sigmoid(const tflite::RuntimeShape &data_shape, int16_t *data)
{
  tflite::reference_integer_ops::Logistic(0 /*data->input_multiplier*/,
                                          0 /*data->input_left_shift */,
                                          data_shape.FlatSize() /*NumElements(input->dims)*/,
                                          data /* tflite::micro::GetTensorData<int16_t>(input) */,
                                          data /*tflite::micro::GetTensorData<int16_t>(output) */);
}

void clipping(const int v_size, const luci_interpreter::lstm::CellStateInfo *cell_state_info,
              int16_t *vector)
{
  for (int i = 0; i < v_size; i++)
  {
    vector[i] = std::max(std::min(cell_state_info->quantized_cell_clip, vector[i]),
                         static_cast<int16_t>(-cell_state_info->quantized_cell_clip));
  }
}
#endif // DIS_QUANT

#ifndef DIS_FLOAT
void fullyConnected(const tflite::FullyConnectedParams &params,
                    const tflite::RuntimeShape &input_shape, const float *input_data,
                    const tflite::RuntimeShape &filter_shape, const float *filter_data,
                    const tflite::RuntimeShape &bias_shape, const float *bias_data,
                    const tflite::RuntimeShape &output_shape, float *output_data)
{
  return tflite::reference_ops::FullyConnected(params, input_shape, input_data, filter_shape,
                                               filter_data, bias_shape, bias_data, output_shape,
                                               output_data);
}

// Input and output have the same shape in LSTM
void mul(const tflite::RuntimeShape &shape, const luci_interpreter::lstm::ArithmeticParams *params,
         const float *input1_data, const float *input2_data, float *output_data)
{
  return mul(params, shape, input1_data, shape, input2_data, shape, output_data);
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

void tanh(int32_t cell_state_scale_power, const tflite::RuntimeShape &input_data_shape,
          float *input_data, const tflite::RuntimeShape &output_data_shape, float *output_data)
{
  tflite::reference_ops::Tanh(input_data_shape, input_data, output_data_shape, output_data);
}

void sigmoid(const tflite::RuntimeShape &data_shape, float *data)
{
  tflite::reference_ops::Logistic(data_shape, data, data_shape, data);
}

void clipping(const int v_size, const luci_interpreter::lstm::CellStateInfo *cell_state_info,
              float *vector)
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
    TFLITE_DCHECK_LE(current_batch_, size_info_.batch_size);
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

  tflite::RuntimeShape inputShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.input_dimension};
    const int32_t *dims_data = reinterpret_cast<const int32_t *>(dims);
    return tflite::RuntimeShape(2, dims_data);
  }

  tflite::RuntimeShape stateShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.state_dimension};
    const int32_t *dims_data = reinterpret_cast<const int32_t *>(dims);
    return tflite::RuntimeShape(2, dims_data);
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
void calculateLstmGate(const LstmStepManager *step_info,
                       const luci_interpreter::lstm::GateParameters *gate_params,
                       // Input FC
                       ActivationType *input_data, const circle::Tensor *input_weight,
                       const circle::Tensor *input_bias,
                       // Recurrent FC
                       ActivationType *recurrent_data, const circle::Tensor *recurrent_weight,
                       const circle::Tensor *recurrent_bias,
                       // Output
                       CellType *gate_output,
                       // Scratch arrays
                       CellType *fc_output_buffer, const TfLiteFusedActivation activation,
                       luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  // Input FC
  const auto gate_output_shape = step_info->stateShape();
  {
    tflite::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params->input_fc_params.input_offset;
    op_params.weights_offset = gate_params->input_fc_params.weights_offset;
    op_params.output_offset = gate_params->input_fc_params.output_offset;
    op_params.output_multiplier = gate_params->input_fc_params.output_multiplier;
    op_params.output_shift = gate_params->input_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params->input_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params->input_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params->input_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params->input_fc_params.float_activation_min;

    fullyConnected(op_params, step_info->inputShape(), input_data + step_info->inputOffset(),
                   luci_interpreter::kernels::getTensorShape(input_weight),
                   luci_interpreter::kernels::getTensorData<WeightType>(
                     runtime_graph->getConstDataByTensor(input_weight)),
                   luci_interpreter::kernels::getTensorShape(input_bias),
                   luci_interpreter::kernels::getTensorData<BiasType>(
                     runtime_graph->getConstDataByTensor(input_bias)),
                   gate_output_shape, gate_output);
  }

  // Recurrent FC
  {
    tflite::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params->recurrent_fc_params.input_offset;
    op_params.weights_offset = gate_params->recurrent_fc_params.weights_offset;
    op_params.output_offset = gate_params->recurrent_fc_params.output_offset;
    op_params.output_multiplier = gate_params->recurrent_fc_params.output_multiplier;
    op_params.output_shift = gate_params->recurrent_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params->recurrent_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params->recurrent_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params->recurrent_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params->recurrent_fc_params.float_activation_min;

    fullyConnected(op_params, step_info->stateShape(),
                   recurrent_data + step_info->hiddenStateOffset(),
                   luci_interpreter::kernels::getTensorShape(recurrent_weight),
                   luci_interpreter::kernels::getTensorData<WeightType>(
                     runtime_graph->getConstDataByTensor(recurrent_weight)),
                   luci_interpreter::kernels::getTensorShape(recurrent_bias),
                   luci_interpreter::kernels::getTensorData<BiasType>(
                     runtime_graph->getConstDataByTensor(recurrent_bias)),
                   gate_output_shape, fc_output_buffer);

    addElementWise(gate_output, fc_output_buffer, /*n_batch=*/gate_output_shape.DimsData()[0],
                   /*n_state=*/gate_output_shape.DimsData()[1], gate_output);

    switch (activation)
    {
      case TfLiteFusedActivation::kTfLiteActSigmoid:
        sigmoid(gate_output_shape, gate_output);
        break;
      case TfLiteFusedActivation::kTfLiteActTanh:
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
}

// Update the hidden state of the LSTM kernel using the following formula:
// updated_hidden_state = Tanh(updated_cell_state) * output_gate_output, * means
// element wise multiplication
template <typename CellType, typename ActivationType>
void updateLstmHidden(const LstmStepManager *step_info, CellType *cell_state_data_base,
                      ActivationType *hidden_state_data, const CellType *output_gate_output,
                      const luci_interpreter::lstm::ArithmeticParams *mul_params,
                      int32_t cell_state_scale_power, CellType *buffer)
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
                    const luci_interpreter::lstm::ArithmeticParams &forget_cell_mul_params,
                    const luci_interpreter::lstm::ArithmeticParams &input_mul_params,
                    const luci_interpreter::lstm::CellStateInfo *cell_state_info, CellType *buffer)
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
                 /*n_batch=*/cell_state_shape.DimsData()[0],
                 /*n_state=*/cell_state_shape.DimsData()[1],
                 cell_state_data + step_info->cellStateOffset());

  if (cell_state_info->cell_clip > 0)
  {
    clipping(cell_state_shape.FlatSize(), cell_state_info,
             cell_state_data + step_info->cellStateOffset());
  }
}

template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
void lstmStep(luci_interpreter::lstm::LSTMStruct *lstm_struct,
              luci_interpreter::lstm::LSTMParameters *lstm_params, LstmStepManager *step_info,
              luci_interpreter::lstm::CellStateInfo *cell_state_info,
              ActivationType *output_state_data, CellType *cell_state_data, CellType *scratch0,
              CellType *scratch1, CellType *scratch2, CellType *scratch3,
              luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  /*Step1: Calculate gate outputs to prepare cell state update*/
  CellType *gate_internal_buffer = scratch3;
  CellType *forget_gate_output = scratch0;

  auto input_data = luci_interpreter::kernels::getTensorData<ActivationType>(
    runtime_graph->getDataByTensor(lstm_struct->input()));

  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->forget_gate_parameters,
    // Input FC
    input_data, lstm_struct->input_to_forget_weights(), lstm_struct->forget_gate_bias(),
    // Recurrent FC
    output_state_data, lstm_struct->recurrent_to_forget_weights(), nullptr,
    // Output
    forget_gate_output, gate_internal_buffer, TfLiteFusedActivation::kTfLiteActSigmoid,
    runtime_graph);

  // Input Gate calculation;
  CellType *input_gate_output = scratch1;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->input_gate_parameters,
    // Input FC
    input_data, lstm_struct->input_to_input_weights(), lstm_struct->input_gate_bias(),
    // Recurrent FC
    output_state_data, lstm_struct->recurrent_to_input_weights(),
    /*recurrent_bias*/ nullptr,
    // Output
    input_gate_output,
    // Scratch arrays
    gate_internal_buffer, TfLiteFusedActivation::kTfLiteActSigmoid, runtime_graph);

  // Cell Gate calculation
  CellType *cell_gate_output = scratch2;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, &lstm_params->cell_gate_parameters,
    // Input FC
    input_data, lstm_struct->input_to_cell_weights(), lstm_struct->cell_gate_bias(),
    // Recurrent FC
    output_state_data, lstm_struct->recurrent_to_cell_weights(),
    /*recurrent_bias*/ nullptr,
    // Output
    cell_gate_output,
    // Scratch arrays
    gate_internal_buffer, TfLiteFusedActivation::kTfLiteActTanh, runtime_graph);

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
      input_data, lstm_struct->input_to_output_weights(), lstm_struct->output_gate_bias(),
      // Recurrent FC
      output_state_data, lstm_struct->recurrent_to_output_weights(), nullptr,
      // Output
      output_gate_output,
      // Scratch arrays
      gate_internal_buffer, TfLiteFusedActivation::kTfLiteActSigmoid, runtime_graph);
    CellType *tanh_activated_cell_buffer = scratch0; // reuse buffer
    updateLstmHidden<CellType, ActivationType>(
      step_info, cell_state_data, output_state_data, output_gate_output,
      &lstm_params->inter_gate_parameters.output_mul_params,
      cell_state_info->cell_state_scale_power, tanh_activated_cell_buffer);

    ActivationType *output_ptr = luci_interpreter::kernels::getTensorData<ActivationType>(
      runtime_graph->getDataByTensor(lstm_struct->output()));
    std::memcpy(output_ptr + step_info->outputOffset(),
                output_state_data + step_info->hiddenStateOffset(),
                step_info->stateShape().FlatSize() * sizeof(ActivationType));
  }
}

} // namespace lstm_internal

// Evaluate the LSTM kernel with (potential) multi-steps and multi-batch input
template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
void evalLSTM(luci_interpreter::lstm::LSTMStruct *lstm_struct,
              luci_interpreter::lstm::LSTMParameters *lstm_params,
              luci_interpreter::lstm::CellStateInfo *cell_state_info,
              ActivationType *output_state_data, CellType *cell_state_data, CellType *scratch0,
              CellType *scratch1, CellType *scratch2, CellType *scratch3,
              luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  lstm_internal::LstmSizeInfo size_info;

  size_info.time_major = lstm_struct->options->time_major();
  size_info.batch_size = size_info.time_major
                           ? luci_interpreter::Tensor::dim(lstm_struct->input(), 1)
                           : luci_interpreter::Tensor::dim(lstm_struct->input(), 0);
  size_info.time_steps = size_info.time_major
                           ? luci_interpreter::Tensor::dim(lstm_struct->input(), 0)
                           : luci_interpreter::Tensor::dim(lstm_struct->input(), 1);
  size_info.input_dimension = luci_interpreter::Tensor::dim(lstm_struct->input(), 2);
  size_info.state_dimension = luci_interpreter::Tensor::dim(lstm_struct->output_state(), 1);

  lstm_internal::LstmStepManager step_info(size_info);

  // time is the first dimention, enable batch computation
  if (size_info.time_major)
  {
    for (int t = 0; t < size_info.time_steps; t++)
    {
      lstm_internal::lstmStep<ActivationType, WeightType, CellType, BiasType>(
        lstm_struct, lstm_params, &step_info, cell_state_info, output_state_data, cell_state_data,
        scratch0, scratch1, scratch2, scratch3, runtime_graph);
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
          scratch0, scratch1, scratch2, scratch3, runtime_graph);
        // prepare for the next time step
        step_info.updateTime();
      }
      // prepare for the next batch
      step_info.updateBatch();
      step_info.resetTime();
    }
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
