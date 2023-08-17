/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_interpreter/core/TrainableWeightStorage.h"

namespace luci_interpreter
{
namespace training
{

Status TrainableWeightStorage::createTrainableWeightForTensor(const circle::Tensor *tensor,
                                                              SimpleMemoryManager *memoryManager,
                                                              const uint8_t *const_data)
{
  assert(_tensor_to_data.count(tensor) == 0 && "Double training weight");

  if (_tensor_to_data.count(tensor) != 0)
  {
    return Error;
  }

  uint8_t *allocated_data = memoryManager->allocate_memory(tensor);

  std::memcpy(allocated_data, const_data,
              size(Tensor::element_type(tensor)) * Tensor::num_elements(tensor));

  _tensor_to_data[tensor] = allocated_data;

  return Ok;
}

training::Status
TrainableWeightStorage::fillTrainableWeightsStorage(const CircleReader *reader,
                                                    SimpleMemoryManager *memory_manager,
                                                    uint32_t number_of_last_trainable_layers)
{
  const auto operators_size = reader->operators().size();
  const auto operators = reader->operators();

  const uint32_t first_trainable_layer_pos = operators_size - number_of_last_trainable_layers;

  for (uint32_t i = first_trainable_layer_pos; i < operators_size; ++i)
  {
    const auto op = operators.at(i);
    assert(op != nullptr);

    const auto *op_inputs = op->inputs();

    for (const int32_t input_idx : *op_inputs)
    {
      if (input_idx == -1)
        continue;
      const circle::Tensor *tensor = reader->tensors()[input_idx];

      if (_tensor_to_data.count(tensor) > 0)
        continue;

      const auto tensor_data = reader->buffers()[tensor->buffer()]->data();
      if (tensor_data != nullptr)
      {
        if (createTrainableWeightForTensor(tensor, memory_manager, tensor_data->data()) ==
            training::Error)
          return training::Error;
      }
    }
  }
  return training::Ok;
}

Status TrainableWeightStorage::clearAllTrainableWeights()
{
  for (const auto &pair : _tensor_to_data)
  {
    delete[] pair.second;
  }

  _tensor_to_data.clear();
  return Ok;
}

Status TrainableWeightStorage::getTrainWeightDataByTensor(const circle::Tensor *tensor,
                                                          uint8_t **result_data)
{
  assert(tensor != nullptr); // CALLER SIDE

  auto it = _tensor_to_data.find(tensor);

  assert(it != _tensor_to_data.end() && "No data");
  if (it == _tensor_to_data.end())
  {
    return Error;
  }

  *result_data = it->second;

  return Ok;
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
