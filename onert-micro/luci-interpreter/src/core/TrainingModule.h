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

#ifndef LUCI_INTERPRETER_CORE_TRAINING_MODULE_H
#define LUCI_INTERPRETER_CORE_TRAINING_MODULE_H

#include "core/RuntimeModule.h"

#include "luci_interpreter/core/TrainableWeightStorage.h"
#include "TrainingGraph.h"

namespace luci_interpreter
{
namespace training
{

class TrainingModule
{
public:
  TrainingModule(RuntimeModule *runtime_module) : _runtime_module(runtime_module)
  {
    // Do nothing
  }

  training::Status enableTrainingMode(training::TrainingSettings &settings,
                                      SimpleMemoryManager *memoryManager);

  training::Status disableTrainingMode(bool resetWeights);

  training::Status computeGradients(const TrainingSettings &settings,
                                    const uint8_t *label_train_data)
  {
    return _training_graph->computeGradients(settings, _runtime_module->_storage.get(),
                                             &_runtime_module->_circle_reader, label_train_data);
  }

  training::Status updateWeights(const TrainingSettings &settings)
  {
    return _training_graph->updateWeights(settings, _runtime_module->_storage.get(),
                                          &_runtime_module->_circle_reader);
  }

private:
  RuntimeModule *_runtime_module;

  std::unique_ptr<training::TrainingGraph> _training_graph;
};

} // namespace training
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_TRAINING_MODULE_H

#endif // ENABLE_TRAINING
