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

#include "core/train/OMTrainingStorage.h"

#include <cassert>

using namespace onert_micro;
using namespace onert_micro::core;
using namespace onert_micro::core::train;

OMStatus OMTrainingStorage::initTrainingStorage(reader::OMCircleReader *reader, const OMConfig &config)
{
  assert(reader != nullptr);
  if (reader == nullptr)
    return UnknownError;

  _training_configs = config.train_config;

  // Reads mapping table
  _backprop_indexes_to_main_indexes_table = reader->readTensorsTrainIndexesTable();

  // Find targets inputs indexes
  {
    const auto inputs_indexes = reader->inputs();
    for (uint32_t i = 0; i < inputs_indexes->size(); ++i)
    {
      const auto input_index = inputs_indexes->operator[](i);
      if (_backprop_indexes_to_main_indexes_table.find(input_index) == _backprop_indexes_to_main_indexes_table.end())
        _targets_indexes.push_back(input_index);
    }
  }

  return Ok;
}

OMStatus OMTrainingStorage::setGradientDataToTensorIndex(uint16_t tensor_index, uint8_t *data)
{
  auto it = _tensor_to_gradients.find(tensor_index);

  assert(it == _tensor_to_gradients.end());
  assert(data != nullptr);

  if (it != _tensor_to_gradients.end())
    return UnknownError;

  _tensor_to_gradients[tensor_index] = data;

  return Ok;
}

uint8_t * OMTrainingStorage::getGradientDataByTensorIndex(uint16_t tensor_index)
{
  auto it = _tensor_to_gradients.find(tensor_index);

  if (it == _tensor_to_gradients.end())
    return nullptr;

  return _tensor_to_gradients.at(tensor_index);
}
