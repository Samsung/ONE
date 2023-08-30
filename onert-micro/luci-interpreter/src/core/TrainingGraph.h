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

#ifndef LUCI_INTERPRETER_SRC_CORE_TRAINING_GRAPH_H
#define LUCI_INTERPRETER_SRC_CORE_TRAINING_GRAPH_H

#include "luci_interpreter/TrainingSettings.h"
#include "luci_interpreter/core/TrainableWeightStorage.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"
#include "memory_managers/SimpleMemoryManager.h"

#include "GradientCalculationStorage.h"

#include <unordered_map>

namespace luci_interpreter
{
namespace training
{

class TrainingGraph
{
public:
  TrainingGraph() = default;

public:
  Status computeGradients(const TrainingSettings &settings, TrainableWeightStorage *storage,
                          CircleReader *reader, const uint8_t *label_train_data);

  Status updateWeights(const TrainingSettings &settings, TrainableWeightStorage *storage,
                       CircleReader *reader);

  GradientCalculationStorage *getGradientCalculationStorage()
  {
    return &_gradient_calculation_storage;
  }

private:
  Status saveLabelDataAsBackDerivative(CircleReader *reader, TrainableWeightStorage *storage,
                                       const uint8_t *label_train_data);

  GradientCalculationStorage _gradient_calculation_storage;
};

} // namespace training
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_SRC_CORE_TRAINING_GRAPH_H

#endif // ENABLE_TRAINING
