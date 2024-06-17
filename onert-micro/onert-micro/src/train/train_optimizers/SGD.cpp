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

#include "OMConfig.h"
#include "train/train_optimizers/SGD.h"
#include "core/memory/OMMemoryManager.h"
#include "core/OMRuntimeShape.h"

using namespace onert_micro;
using namespace onert_micro::train;
using namespace onert_micro::train::optimizers;

void SGD::reset()
{
  for (auto &cur_tensor_index_data : _tensor_index_to_gradient)
  {
    uint8_t *allocated_data = cur_tensor_index_data.second;

    core::memory::OMMemoryManager::deallocateMemory(allocated_data);
  }
  _tensor_index_to_gradient.clear();
}

/*
 * Update internal states according to calculated gradients using Adam theory
 * grad(t) = grad(t - 1) + calculated_grad(t)
 */
OMStatus SGD::handle(core::OMRuntimeStorage &backward_storage, core::OMRuntimeContext &context)
{
  auto &backward_tensor_to_data = backward_storage.getTensorIndexToData();
  // Check is allocated or not helper buffers
  if (_tensor_index_to_gradient.empty())
  {
    // If not - let's just move it with calculations
    // Goes over all calculated gradients
    // Warning: assume that backward storage at this moment contains only weigths gradients -
    // This should be done due to execution plan work
    for (auto &tensor_to_data : backward_tensor_to_data)
    {
      // Move data
      _tensor_index_to_gradient[tensor_to_data.first] = tensor_to_data.second;
      tensor_to_data.second = nullptr;
    }
    backward_tensor_to_data.clear();
  }
  else
  {
    // Goes over all calculated gradients
    // Warning: assume that backward storage at this moment contains only weigths gradients -
    // This should be done due to execution plan work
    for (auto &tensor_to_data : backward_tensor_to_data)
    {
      auto tensor = context.getTensorByIndex(tensor_to_data.first);
      core::OMRuntimeShape shape(tensor);

      const auto flat_size = shape.flatSize();

      float *grad_data = reinterpret_cast<float *>(_tensor_index_to_gradient[tensor_to_data.first]);
      float *calculated_data = reinterpret_cast<float *>(tensor_to_data.second);

      for (uint32_t i = 0; i < flat_size; ++i)
      {
        grad_data[i] += calculated_data[i];
      }
    }
  }

  return Ok;
}

/*
 * Update weights according to Adam theory:
 *
 * w(t + 1) = w(t) - lambda * grad(t) / batch_size
 */
OMStatus SGD::updateWeights(const onert_micro::OMTrainingContext &training_config,
                            core::OMRuntimeContext &context)
{
  assert(_tensor_index_to_gradient.size() > 0);
  if (_tensor_index_to_gradient.empty())
    return UnknownError;

  for (auto &tensor_to_data : _tensor_index_to_gradient)
  {
    auto tensor = context.getTensorByIndex(tensor_to_data.first);
    core::OMRuntimeShape shape(tensor);

    const auto flat_size = shape.flatSize();

    float *grad_data = reinterpret_cast<float *>(tensor_to_data.second);
    uint8_t *weight_data = nullptr;
    if (context.getConstDataByTensorIndex(&weight_data, tensor_to_data.first) != Ok)
      return UnknownError;

    assert(weight_data != nullptr);
    if (weight_data == nullptr)
      return UnknownError;

    float *f_weight_data = reinterpret_cast<float *>(weight_data);
    float lambda = training_config.learning_rate;
    const uint32_t batch_size = training_config.batch_size;
    for (uint32_t i = 0; i < flat_size; ++i)
    {
      f_weight_data[i] -= (lambda * grad_data[i]) / (static_cast<float>(batch_size));
    }
  }
  return Ok;
}
