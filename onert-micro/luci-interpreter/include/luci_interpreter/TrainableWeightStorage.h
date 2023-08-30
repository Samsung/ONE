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

#ifndef LUCI_INTERPRETER_CORE_TRAINABLE_WEIGHT_STORAGE_H
#define LUCI_INTERPRETER_CORE_TRAINABLE_WEIGHT_STORAGE_H

#include "luci_interpreter/TrainingSettings.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"
#include "memory_managers/SimpleMemoryManager.h"

#include <unordered_map>

namespace luci_interpreter
{
namespace training
{

class TrainableWeightStorage
{
public:
  TrainableWeightStorage() = default;

public:
  Status getTrainWeightDataByTensor(const circle::Tensor *tensor, uint8_t **result_data);

  Status clearAllTrainableWeights();

  training::Status fillTrainableWeightsStorage(const CircleReader *reader,
                                               SimpleMemoryManager *memory_manager,
                                               uint32_t number_of_last_trainable_layers);

private:
  Status createTrainableWeightForTensor(const circle::Tensor *tensor,
                                        SimpleMemoryManager *memoryManager,
                                        const uint8_t *const_data);

private:
  std::unordered_map<const circle::Tensor *, uint8_t *> _tensor_to_data;
};

} // namespace training
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_TRAINABLE_WEIGHT_STORAGE_H

#endif // ENABLE_TRAINING
