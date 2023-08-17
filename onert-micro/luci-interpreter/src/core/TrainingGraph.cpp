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
  assert(settings.number_of_last_trainable_layers == 1);
  auto last_op_pos = reader->operators().size() - 1;
  const auto last_op = reader->operators().at(last_op_pos);

  const auto opcode = reader->builtin_code(last_op);

  assert(opcode == circle::BuiltinOperator_FULLY_CONNECTED);

  Status status = kernel_train.train_kernel(last_op, opcode, reader, &_gradient_calculation_storage,
                                            settings, storage, label_train_data);

  if (status != Ok)
    return status;

  _gradient_calculation_storage.clearComputedData();

  return Ok;
}

Status TrainingGraph::updateWeights(const TrainingSettings &settings,
                                    TrainableWeightStorage *storage, CircleReader *reader)
{
  assert(settings.number_of_last_trainable_layers == 1);
  auto last_op_pos = reader->operators().size() - 1;
  const auto last_op = reader->operators().at(last_op_pos);

  const auto opcode = reader->builtin_code(last_op);

  assert(opcode == circle::BuiltinOperator_FULLY_CONNECTED);

  Status status = kernel_train.train_kernel(last_op, opcode, reader, &_gradient_calculation_storage,
                                            settings, storage, nullptr);

  assert(status == Ok);

  if (status != Ok)
  {
    return status;
  }

  _gradient_calculation_storage.clearComputedData();
  _gradient_calculation_storage.clearComputedGradients();

  return Ok;
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
