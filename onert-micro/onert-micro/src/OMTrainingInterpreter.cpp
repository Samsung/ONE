/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OMTrainingInterpreter.h"

using namespace onert_micro;

OMStatus OMTrainingInterpreter::import(const char *model_ptr, const char *backpropagation_model_ptr, const OMConfig &config)
{
  return _training_runtime_module.import(model_ptr, backpropagation_model_ptr, config);
}

OMStatus OMTrainingInterpreter::forward() { return _training_runtime_module.forward(); }
OMStatus OMTrainingInterpreter::backward() { return _training_runtime_module.backward(); }

OMStatus OMTrainingInterpreter::reset()
{
  return _training_runtime_module.reset();
}

uint32_t OMTrainingInterpreter::getNumberOfInputs() { return _training_runtime_module.getNumberOfInputs(); }
uint32_t OMTrainingInterpreter::getNumberOfOutputs() { return _training_runtime_module.getNumberOfOutputs(); }
uint32_t OMTrainingInterpreter::getNumberOfTargets() { return _training_runtime_module.getNumberOfTargets(); }

void *OMTrainingInterpreter::getInputDataAt(uint32_t position)
{
  return _training_runtime_module.getInputDataAt(position);
}
void *OMTrainingInterpreter::getOutputDataAt(uint32_t position)
{
  return _training_runtime_module.getOutputDataAt(position);
}
void *OMTrainingInterpreter::getTargetDataAt(uint32_t position)
{
  return _training_runtime_module.getTargetDataAt(position);
}

uint32_t OMTrainingInterpreter::getInputSizeAt(uint32_t position)
{
  return _training_runtime_module.getInputSizeAt(position);
}
uint32_t OMTrainingInterpreter::getOutputSizeAt(uint32_t position)
{
  return _training_runtime_module.getOutputSizeAt(position);
}
uint32_t OMTrainingInterpreter::getTargetSizeAt(uint32_t position)
{
  return _training_runtime_module.getTargetSizeAt(position);
}

OMStatus OMTrainingInterpreter::updateWeights()
{
  return _training_runtime_module.updateWeights();
}

OMStatus OMTrainingInterpreter::allocateInputs() { return _training_runtime_module.allocateInputs(); }
OMStatus OMTrainingInterpreter::allocateTargets() { return _training_runtime_module.allocateTargets(); }
