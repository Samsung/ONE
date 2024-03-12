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
#include "core/memory/OMMemoryManager.h"
#include "core/OMRuntimeShape.h"
#include "core/OMDataType.h"

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

  // Allocate buffers for exponent average squares values if needed
  if (config.train_config.optimization_strategy == RMSProp or config.train_config.optimization_strategy == ADAM)
  {
    for (const auto &pair : _backprop_indexes_to_main_indexes_table)
    {
      const auto tensor_index = pair.first;
      const auto tensor = reader->tensors()->operator[](tensor_index);
      const OMRuntimeShape tensor_shape(tensor);

      int32_t num_elements = tensor_shape.flatSize();
      assert(num_elements >= 0 && "Num elements should be positive");
      if (num_elements < 0)
        return UnknownError;
      const auto casted_num_elements = static_cast<uint32_t>(num_elements);
      const auto type_size =
        static_cast<uint32_t>(getOMDataTypeSize(onertMicroDatatype(tensor->type())));

      // allocate data
      uint8_t *allocated_data = nullptr;
      OMStatus status =
        memory::OMMemoryManager::allocateMemory(casted_num_elements * type_size, &allocated_data);
      if (status != Ok)
        return status;

      std::memset(allocated_data, 0.f, casted_num_elements * type_size);

      _tensor_to_exponent_avg_squares[tensor_index] = allocated_data;

      // allocate second buffer for ADAM
      if (config.train_config.optimization_strategy == ADAM)
      {
        // allocate data
        allocated_data = nullptr;
        status =
          memory::OMMemoryManager::allocateMemory(casted_num_elements * type_size, &allocated_data);
        if (status != Ok)
          return status;

        std::memset(allocated_data, 0.f, casted_num_elements * type_size);

        _tensor_to_exponent_avg[tensor_index] = allocated_data;
      }
    }
  }

  return Ok;
}

OMTrainingStorage::~OMTrainingStorage()
{
  for (const auto &pair : _tensor_to_exponent_avg_squares)
  {
    uint8_t *allocated_data = pair.second;
    memory::OMMemoryManager::deallocateMemory(allocated_data);
  }
  _tensor_to_exponent_avg_squares.clear();

  for (const auto &pair : _tensor_to_exponent_avg)
  {
    uint8_t *allocated_data = pair.second;
    memory::OMMemoryManager::deallocateMemory(allocated_data);
  }
  _tensor_to_exponent_avg.clear();
}

uint8_t *OMTrainingStorage::getExponentAvgSquaresData(uint16_t tensor_index)
{
  auto it = _tensor_to_exponent_avg_squares.find(tensor_index);

  if (it == _tensor_to_exponent_avg_squares.end())
    return nullptr;

  return it->second;
}

uint8_t *OMTrainingStorage::getExponentAvgData(uint16_t tensor_index)
{
  auto it = _tensor_to_exponent_avg.find(tensor_index);

  if (it == _tensor_to_exponent_avg.end())
    return nullptr;

  return it->second;
}

//
//OMStatus OMTrainingStorage::setGradientDataToTensorIndex(uint16_t tensor_index, uint8_t *data)
//{
//  auto it = _tensor_to_gradients.find(tensor_index);
//
//  assert(it == _tensor_to_gradients.end());
//  assert(data != nullptr);
//
//  if (it != _tensor_to_gradients.end())
//    return UnknownError;
//
//  _tensor_to_gradients[tensor_index] = data;
//
//  return Ok;
//}
//
//uint8_t * OMTrainingStorage::getGradientDataByTensorIndex(uint16_t tensor_index)
//{
//  auto it = _tensor_to_gradients.find(tensor_index);
//
//  if (it == _tensor_to_gradients.end())
//    return nullptr;
//
//  return _tensor_to_gradients.at(tensor_index);
//}
