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

#include "import/OMKernelConfigureBuilder.h"
#include "core/OMUtils.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMShape.h"

#include "PALUnidirectionalSequenceLSTMKernel.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t input2TensorIdx = 1;
constexpr uint32_t outputTensorIdx = 0;

OMStatus validateWeightTensorSize(const core::OMRuntimeShape &weight_shape, int dim1_size,
                                  int dim2_size)
{
  if (weight_shape.dimensionsCount() != 2 or weight_shape.dims(0) != dim1_size or
      weight_shape.dims(1) != dim2_size)
    return FailedCheckCondition;

  return Ok;
}

OMStatus validateTensorsSize(execute::lstm::LSTMStruct *lstm_struct, const bool time_major)
{
  OMStatus status = Ok;

  OMRuntimeShape input_shape(lstm_struct->input());
  OMRuntimeShape output_state_shape(lstm_struct->output_state());
  OMRuntimeShape cell_state_shape(lstm_struct->cell_state());
  OMRuntimeShape output_shape(lstm_struct->output());

  const auto batch_size = time_major ? input_shape.dims(1) : input_shape.dims(0);

  const auto input_dimension = input_shape.dims(2);
  const auto state_dimension =
    output_state_shape.dims(1); // Tensor::dim(lstm_struct->output_state(), 1);

  // Input FC weights
  for (int32_t i = 1; i < 5; i++)
  {
    status = validateWeightTensorSize(lstm_struct->get_internal_tensor(i), state_dimension,
                                      input_dimension);
    if (status != Ok)
      return status;
  }

  // Recurrent FC weights
  for (int32_t i = 5; i < 9; i++)
  {
    status = validateWeightTensorSize(lstm_struct->get_internal_tensor(i), state_dimension,
                                      state_dimension);
    if (status != Ok)
      return status;
  }

  // Biases
  for (int32_t i = 12; i < 16; i++)
  {
    OMRuntimeShape shape(lstm_struct->get_internal_tensor(i));
    if (shape.dimensionsCount() != 1 or shape.dims(0) != state_dimension)
      return FailedCheckCondition;
  }

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  if (output_state_shape.flatSize() != batch_size * state_dimension or
      cell_state_shape.flatSize() != batch_size * state_dimension)
    return FailedCheckCondition;

  // Check the shape of output tensor against that of input tensor
  if (output_shape.dimensionsCount() != 3 or input_shape.dims(0) != output_shape.dims(0) or
      input_shape.dims(1) != output_shape.dims(1) or output_shape.dims(2) != state_dimension)
    return FailedCheckCondition;

  return Ok;
}

} // namespace

OMStatus onert_micro::import::configure_kernel_CircleUnidirectionalSequenceLSTM(
  const OMConfigureArgs &config_args)
{
  OMRuntimeContext &runtime_context = config_args.runtime_context;
  uint16_t op_index = config_args.kernel_index;
  OMRuntimeStorage &runtime_storage = config_args.runtime_storage;

  OMStatus status = Ok;

  execute::lstm::LSTMStruct lstm_struct{};

  status = lstm_struct.readKernel(op_index, runtime_storage, runtime_context);
  if (status != Ok)
    return status;

  if (lstm_struct.input()->type() != circle::TensorType_FLOAT32 and
      lstm_struct.input()->type() != circle::TensorType_INT8)
    return UnsupportedType;

  status = lstm_struct.validateTensorTypes();
  if (status != Ok)
    return status;

  const bool time_major = lstm_struct.options->time_major();

  status = validateTensorsSize(&lstm_struct, time_major);
  if (status != Ok)
    return status;

  // No peephole
  for (int32_t i = 9; i < 12; ++i)
    if (lstm_struct.get_internal_tensor(i) != nullptr)
      return FailedCheckCondition;

  // No projection
  for (int32_t i = 16; i < 18; ++i)
    if (lstm_struct.get_internal_tensor(i) != nullptr)
      return FailedCheckCondition;

  // No internal layer norm
  for (int32_t i = 20; i < 24; ++i)
    if (lstm_struct.get_internal_tensor(i) != nullptr)
      return FailedCheckCondition;

  return status;
}
