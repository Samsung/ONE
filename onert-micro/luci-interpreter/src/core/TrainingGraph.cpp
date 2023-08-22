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

#ifdef ENABLE_TRAINING

#include "TrainingGraph.h"

#include "kernels/KernelBuilder.h"

#include <unordered_map>

namespace luci_interpreter
{
namespace training
{

Status TrainingGraph::computeGradients(const TrainingSettings &settings,
                                       TrainableWeightStorage *storage, CircleReader *reader,
                                       const uint8_t *label_train_data)
{
  auto last_op_pos = reader->operators().size() - 1;
  uint8_t *gradients_values = nullptr;
  uint8_t *gradients_values_prev = nullptr;

  Status status;

  for (auto op_pos = last_op_pos; op_pos >= 0; --op_pos)
  {
    const auto op = reader->operators().at(op_pos);
    const auto opcode = reader->builtin_code(op);
    assert(opcode == circle::BuiltinOperator_FULLY_CONNECTED);

    TrainingSettings settings_tmp = settings;

    gradients_values_prev = gradients_values;
    const auto weight_index = op->inputs()->operator[](1);
    assert(weight_index != -1);
    const auto weights = reader->tensors()[weight_index];
    assert(weights != nullptr);

    const auto rows = Tensor::dim(weights, 0);
    const auto cols = Tensor::dim(weights, 1);

    status = _gradient_calculation_storage.getGradients(weights, &gradients_values);
    float *gradient_values_float = reinterpret_cast<float *>(gradients_values);
    assert(gradient_values_float != nullptr);

    if (op_pos == last_op_pos)
    {
      settings_tmp.is_last_layer = true;
      status = kernel_train.train_kernel(op, opcode, reader, &_gradient_calculation_storage,
                                         settings_tmp, storage, label_train_data);
    }
    else
    {
      settings_tmp.is_last_layer = false;
      status = kernel_train.train_kernel(op, opcode, reader, &_gradient_calculation_storage,
                                         settings_tmp, storage, gradients_values_prev);
    }
    if (status != Ok)
      return status;
  }

  _gradient_calculation_storage.clearComputedData();

  return Ok;
}

Status TrainingGraph::updateWeights(const TrainingSettings &settings,
                                    TrainableWeightStorage *storage, CircleReader *reader)
{
  auto last_op_pos = reader->operators().size() - 1;
  Status status;
  for (auto op_pos = last_op_pos; op_pos >= 0; --op_pos)
  {
    const auto op = reader->operators().at(op_pos);
    const auto opcode = reader->builtin_code(op);
    assert(opcode == circle::BuiltinOperator_FULLY_CONNECTED);

    status = kernel_train.train_kernel(op, opcode, reader, &_gradient_calculation_storage, settings,
                                       storage, nullptr);

    assert(status == Ok);

    if (status != Ok)
    {
      return status;
    }
  }
  _gradient_calculation_storage.clearComputedData();
  _gradient_calculation_storage.clearComputedGradients();

  return Ok;
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
