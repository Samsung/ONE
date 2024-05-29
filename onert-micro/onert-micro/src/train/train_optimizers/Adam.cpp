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
#include "train/train_optimizers/Adam.h"
#include "core/memory/OMMemoryManager.h"
#include "core/OMRuntimeShape.h"
#include "core/OMDataType.h"

#include <cmath>

using namespace onert_micro;
using namespace onert_micro::train;
using namespace onert_micro::train::optimizers;

void Adam::reset()
{
  for (auto &cur_tensor_index_data : _tensor_to_exponent_avg)
  {
    uint8_t *allocated_data = cur_tensor_index_data.second;

    core::memory::OMMemoryManager::deallocateMemory(allocated_data);
  }
  _tensor_to_exponent_avg.clear();

  for (auto &cur_tensor_index_data : _tensor_to_exponent_avg_squares)
  {
    uint8_t *allocated_data = cur_tensor_index_data.second;

    core::memory::OMMemoryManager::deallocateMemory(allocated_data);
  }
  _tensor_to_exponent_avg_squares.clear();
}

/*
 * Update internal states according to calculated gradients using Adam theory
 * m(t) = beta_1 * m(t-1) + (1 - beta_1) * calculated_gradients(t)
 * v(t) = beta_2 * v(t-1) + (1 - beta_2) * (calculated_gradients(t)) ^ 2
 */
OMStatus Adam::handle(const OMTrainingContext &training_config,
                      core::OMRuntimeStorage &backward_storage, core::OMRuntimeContext &context)
{
  auto &backward_tensor_to_data = backward_storage.getTensorIndexToData();

  // Check is allocated or not helper buffers
  if (_tensor_to_exponent_avg_squares.empty())
  {
    // If not - let's allocate it
    assert(_tensor_to_exponent_avg.empty() == true);
    // Goes over all calculated gradients
    // Warning: assume that backward storage at this moment contains only weigths gradients -
    // This should be done due to execution plan work
    for (auto &tensor_to_data : backward_tensor_to_data)
    {
      auto tensor = context.getTensorByIndex(tensor_to_data.first);
      core::OMRuntimeShape shape(tensor);

      const auto flat_size = shape.flatSize();
      const auto type_size = sizeof(core::OMDataType(tensor->type()));

      // Allocate data for exponent calculation
      uint8_t *exponent_data = nullptr;
      OMStatus status =
        core::memory::OMMemoryManager::allocateMemory(flat_size * type_size, &exponent_data);
      assert(status == Ok);
      if (status != Ok)
        return UnknownError;
      // Set to zeros
      std::memset(exponent_data, 0, flat_size * type_size);
      _tensor_to_exponent_avg[tensor_to_data.first] = exponent_data;

      // Allocate data for exponent square calculation
      uint8_t *exponent_square_data = nullptr;
      status =
        core::memory::OMMemoryManager::allocateMemory(flat_size * type_size, &exponent_square_data);
      assert(status == Ok);
      if (status != Ok)
        return UnknownError;
      // Set to zeros
      std::memset(exponent_square_data, 0, flat_size * type_size);
      _tensor_to_exponent_avg_squares[tensor_to_data.first] = exponent_square_data;
    }
  }

  // Update state
  // Goes over all calculated gradients
  // Warning: assume that backward storage at this moment contains only weigths gradients -
  // This should be done due to execution plan work
  for (auto &tensor_to_data : backward_tensor_to_data)
  {
    auto tensor = context.getTensorByIndex(tensor_to_data.first);
    core::OMRuntimeShape shape(tensor);

    const auto flat_size = shape.flatSize();

    float *exponent_data = reinterpret_cast<float *>(_tensor_to_exponent_avg[tensor_to_data.first]);
    float *exponent_square_data =
      reinterpret_cast<float *>(_tensor_to_exponent_avg_squares[tensor_to_data.first]);
    float *calculated_data = reinterpret_cast<float *>(tensor_to_data.second);
    float beta = training_config.beta;
    float beta_squares = training_config.beta_squares;
    for (uint32_t i = 0; i < flat_size; ++i)
    {
      exponent_data[i] = beta * exponent_data[i] + (1 - beta) * calculated_data[i];
      exponent_square_data[i] = beta_squares * exponent_square_data[i] +
                                (1 - beta_squares) * std::pow(calculated_data[i], 2);
    }
  }

  return Ok;
}

/*
 * Update weights according to Adam theory:
 * m`(t) = m(t) / (1 - (beta_1) ^ t)
 * v`(t) = v(t) / (1 - (beta_2) ^ t)
 *
 * w(t + 1) = w(t) - lambda * m`(t) / (sqrt(v` + epsilon))
 */
OMStatus Adam::updateWeights(const onert_micro::OMTrainingContext &training_config,
                             core::OMRuntimeContext &context)
{
  assert(_tensor_to_exponent_avg.size() > 0);
  assert(_tensor_to_exponent_avg_squares.size() > 0);

  if (_tensor_to_exponent_avg_squares.empty() or _tensor_to_exponent_avg.empty() or
      _tensor_to_exponent_avg.size() != _tensor_to_exponent_avg_squares.size())
  {
    return UnknownError;
  }

  for (auto &tensor_to_data : _tensor_to_exponent_avg)
  {
    auto exponent_squares_it = _tensor_to_exponent_avg_squares.find(tensor_to_data.first);
    if (exponent_squares_it == _tensor_to_exponent_avg_squares.end())
      return UnknownError;

    auto tensor = context.getTensorByIndex(tensor_to_data.first);
    core::OMRuntimeShape shape(tensor);

    const auto flat_size = shape.flatSize();

    float *exponent_data = reinterpret_cast<float *>(tensor_to_data.second);
    float *exponent_square_data = reinterpret_cast<float *>(exponent_squares_it->second);

    uint8_t *weight_data = nullptr;
    if (context.getConstDataByTensorIndex(&weight_data, tensor_to_data.first) != Ok)
      return UnknownError;

    assert(weight_data != nullptr);
    if (weight_data == nullptr)
      return UnknownError;

    float *f_weight_data = reinterpret_cast<float *>(weight_data);
    float lambda = training_config.lambda;
    float beta = training_config.beta;
    float beta_squares = training_config.beta_squares;
    float batches = static_cast<float>(training_config.batch_size);
    float num_step = static_cast<float>(training_config.num_step);
    float beta_in_pow_batch = std::pow(beta, num_step);
    float beta_square_in_pow_batch = std::pow(beta_squares, num_step);
    float epsilon = training_config.epsilon;
    for (uint32_t i = 0; i < flat_size; ++i)
    {
      float exponent_corrected = exponent_data[i] / (1.f - beta_in_pow_batch);
      float exponent_square_corrected = exponent_square_data[i] / (1.f - beta_square_in_pow_batch);
      f_weight_data[i] -=
        lambda * (exponent_corrected / (std::sqrt(exponent_square_corrected + epsilon)));
    }
  }

  return Ok;
}
