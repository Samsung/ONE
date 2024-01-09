/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMUtils.h"
#include "execute/OMUtils.h"
#include "core/OMShape.h"
#include "core/OMDataType.h"
#include "PALUnidirectionalSequenceLSTM.h"
#include "PALUnidirectionalSequenceLSTMKernel.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace

#ifndef DIS_FLOAT
lstm::FullyConnectedParams createFcParamsFloat()
{
  lstm::FullyConnectedParams op_params{};
  calculateActivationRange(circle::ActivationFunctionType_NONE, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  op_params.quantized_activation_max = op_params.float_activation_max;
  op_params.quantized_activation_min = op_params.float_activation_min;
  return op_params;
}

lstm::GateParameters createGateParamsFloat()
{
  lstm::GateParameters gate_params{};

  gate_params.input_fc_params = createFcParamsFloat();
  gate_params.recurrent_fc_params = createFcParamsFloat();

  return gate_params;
}

lstm::CellStateInfo createLstmCellStateInfoFloat(const float cell_clip)
{
  lstm::CellStateInfo cell_state_info{};
  cell_state_info.cell_clip = cell_clip;
  cell_state_info.cell_state_scale_power = 0; // no quantization
  cell_state_info.quantized_cell_clip = 0;    // no quantization
  return cell_state_info;
}

OMStatus prepareGateParamsFloat(lstm::LSTMParameters *float_lstm_params)
{
  // Gate Parameters
  float_lstm_params->forget_gate_parameters = createGateParamsFloat();
  float_lstm_params->input_gate_parameters = createGateParamsFloat();
  float_lstm_params->cell_gate_parameters = createGateParamsFloat();
  float_lstm_params->output_gate_parameters = createGateParamsFloat();

  // Inter gate multiplication parameters
  lstm::ArithmeticParams op_params{};
  OMStatus status =
    calculateActivationRange(circle::ActivationFunctionType_NONE, &op_params.float_activation_min,
                             &op_params.float_activation_max);

  op_params.quantized_activation_max = op_params.float_activation_max;
  op_params.quantized_activation_min = op_params.float_activation_min;
  float_lstm_params->inter_gate_parameters.forget_cell_mul_params = op_params;
  float_lstm_params->inter_gate_parameters.input_mul_params = op_params;
  float_lstm_params->inter_gate_parameters.output_mul_params = op_params;

  return status;
}

OMStatus evalFloat(lstm::LSTMStruct &lstm_struct, core::OMRuntimeStorage &storage,
                   core::OMRuntimeContext &context)
{
  lstm::CellStateInfo cell_state_info =
    createLstmCellStateInfoFloat(lstm_struct.options->cell_clip());

  lstm::LSTMParameters lstm_params{};
  OMStatus status = prepareGateParamsFloat(&lstm_params);

  core::OMRuntimeShape input_shape(lstm_struct.input());
  core::OMRuntimeShape output_state_shape(lstm_struct.output_state());
  core::OMRuntimeShape cell_state_shape(lstm_struct.cell_state());

  const bool time_major = lstm_struct.options->time_major();
  const auto batch_size =
    time_major ? input_shape.dims(1)
               : input_shape.dims(
                   0); // Tensor::dim(lstm_struct.input(), 1) : Tensor::dim(lstm_struct.input(), 0);
  const auto state_dimension =
    output_state_shape.dims(1); // Tensor::dim(lstm_struct.output_state(), 1);
  const auto cell_state_type_size = core::getOMDataTypeSize(core::onertMicroDatatype(
    lstm_struct.cell_state()
      ->type())); // getDataTypeSize(Tensor::element_type(lstm_struct.cell_state()));

  auto scratch_0_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_1_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_2_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_3_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);

  // Create and fill with 0 output state tensor
  auto output_state_data = std::make_unique<float[]>(output_state_shape.flatSize());
  std::fill_n(output_state_data.get(), output_state_shape.flatSize(), 0);

  // Create and fill with 0 cell state tensor
  auto cell_state_data = std::make_unique<float[]>(cell_state_shape.flatSize());
  std::fill_n(cell_state_data.get(), cell_state_shape.flatSize(), 0);

  status = pal::evalLSTM<float, float, float, float>(
    &lstm_struct, &lstm_params, &cell_state_info, output_state_data.get(), cell_state_data.get(),
    core::utils::castOutputData<float>(scratch_0_data.get()),
    core::utils::castOutputData<float>(scratch_1_data.get()),
    core::utils::castOutputData<float>(scratch_2_data.get()),
    core::utils::castOutputData<float>(scratch_3_data.get()), storage, context);

  return status;
}
#endif // DIS_FLOAT

// NOTE: doesnt currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleUnidirectionalSequenceLSTM(
  const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  OMStatus status = Ok;

  execute::lstm::LSTMStruct lstm_struct{};

  status = lstm_struct.readKernel(op_index, runtime_storage, runtime_context);
  if (status != Ok)
    return status;

  status = lstm_struct.readData(op_index, runtime_storage, runtime_context);
  if (status != Ok)
    return status;

  switch (lstm_struct.input()->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
      status = evalFloat(lstm_struct, runtime_storage, runtime_context);
      break;
#endif // DIS_FLOAT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
