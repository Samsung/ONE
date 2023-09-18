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

#include "TrainingModule.h"

#include <memory>

namespace luci_interpreter
{
namespace training
{

training::Status TrainingModule::enableTrainingMode(training::TrainingSettings &settings,
                                                    SimpleMemoryManager *memoryManager)
{
  if (_runtime_module->_storage.get() == nullptr)
  {
    _runtime_module->_storage = std::make_unique<TrainableWeightStorage>();
  }

  if (_runtime_module->_storage->fillTrainableWeightsStorage(
        &_runtime_module->_circle_reader, memoryManager,
        settings.number_of_last_trainable_layers) == training::Error)
    return training::Error;

  _training_graph = std::make_unique<training::TrainingGraph>();

  for (auto &graph : _runtime_module->_graphs)
  {
    graph.setLastTrainingLayersNumber(settings.number_of_last_trainable_layers);
    graph.setGradientCalculationStorage(_training_graph->getGradientCalculationStorage());
    graph.setTrainingWeightStorage(_runtime_module->_storage.get());
  }

  return training::Ok;
}

training::Status TrainingModule::disableTrainingMode(bool resetWeights)
{
  _training_graph.release();

  if (resetWeights)
  {
    if (_runtime_module->_storage->clearAllTrainableWeights() == training::Error)
      return training::Error;
    _runtime_module->_storage.release();
  }

  for (auto &graph : _runtime_module->_graphs)
  {
    graph.setLastTrainingLayersNumber(0);
    graph.setGradientCalculationStorage(nullptr);
    if (resetWeights)
      graph.setTrainingWeightStorage(nullptr);
  }

  return training::Ok;
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
