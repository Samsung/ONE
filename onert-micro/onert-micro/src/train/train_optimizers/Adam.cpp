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
OMStatus Adam::handle(core::OMRuntimeStorage &backward_storage, core::OMRuntimeContext &context)
{
  auto &backward_tensor_to_data = backward_storage.getTensorIndexToData();

  // Check is allocated or not helper buffers
  if (_tensor_to_exponent_avg_squares.empty())
  {
    // If not - let's allocate it
    assert(_tensor_to_exponent_avg.empty() == true);
    // Goes over all calculated gradients
    // Warning: assume that backward storage at this moment contains only weighs gradients -
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

  // Check is allocated or not helper buffer
  if (_tensor_index_to_gradient.empty())
  {
    // If not - let's just move it with calculations
    // Goes over all calculated gradients
    // Warning: assume that backward storage at this moment contains only weights gradients -
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
    // Warning: assume that backward storage at this moment contains only weighs gradients -
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
 * Update internal states according to calculated gradients using Adam theory
 * m(t) = beta_1 * m(t-1) + (1 - beta_1) * calculated_gradients(t)
 * v(t) = beta_2 * v(t-1) + (1 - beta_2) * (calculated_gradients(t)) ^ 2

 * Update weights according to Adam theory:
 * m`(t) = m(t) / (1 - (beta_1) ^ t)
 * v`(t) = v(t) / (1 - (beta_2) ^ t)
 *
 * w(t + 1) = w(t) - lambda * m`(t) / (sqrt(v` + epsilon))
 */
OMStatus Adam::updateWeights(const onert_micro::OMTrainingContext &training_config,
                             core::OMRuntimeContext &context)
{
  assert(_tensor_index_to_gradient.size() > 0);

  for (auto &tensor_to_data : _tensor_index_to_gradient)
  {
    auto exponent_squares_it = _tensor_to_exponent_avg_squares.find(tensor_to_data.first);
    if (exponent_squares_it == _tensor_to_exponent_avg_squares.end())
      return UnknownError;

    auto exponent_it = _tensor_to_exponent_avg.find(tensor_to_data.first);
    if (exponent_it == _tensor_to_exponent_avg.end())
      return UnknownError;

    auto tensor = context.getTensorByIndex(tensor_to_data.first);
    core::OMRuntimeShape shape(tensor);

    const auto flat_size = shape.flatSize();

    float *exponent_data = reinterpret_cast<float *>(exponent_it->second);
    float *exponent_square_data = reinterpret_cast<float *>(exponent_squares_it->second);
    float *calculated_data = reinterpret_cast<float *>(tensor_to_data.second);
    float beta = training_config.beta;
    float beta_squares = training_config.beta_squares;
    float batches = static_cast<float>(training_config.batch_size);
    for (uint32_t i = 0; i < flat_size; ++i)
    {
      const auto cur_val = calculated_data[i] / batches;
      exponent_data[i] = beta * exponent_data[i] + (1 - beta) * cur_val;
      exponent_square_data[i] =
        beta_squares * exponent_square_data[i] + (1 - beta_squares) * cur_val * cur_val;
    }

    uint8_t *weight_data = nullptr;
    if (context.getConstDataByTensorIndex(&weight_data, tensor_to_data.first) != Ok)
      return UnknownError;

    assert(weight_data != nullptr);
    if (weight_data == nullptr)
      return UnknownError;

    float *f_weight_data = reinterpret_cast<float *>(weight_data);
    float lambda = training_config.lambda;
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
