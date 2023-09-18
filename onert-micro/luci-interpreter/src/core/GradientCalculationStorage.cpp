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

#include "GradientCalculationStorage.h"
#include "luci_interpreter/core/Tensor.h"

#include <unordered_map>

namespace luci_interpreter
{
namespace training
{

Status GradientCalculationStorage::clearComputedData()
{
  for (const auto &pair : _tensor_to_data)
  {
    delete[] pair.second;
  }

  _tensor_to_data.clear();

  return Ok;
}

Status GradientCalculationStorage::clearComputedGradients()
{
  for (const auto &pair : _gradients)
  {
    delete[] pair.second;
  }

  _gradients.clear();

  return Ok;
}

Status GradientCalculationStorage::saveDataToTensor(const circle::Tensor *tensor, uint8_t *data)
{
  const auto it = _tensor_to_data.find(tensor);
  if (it != _tensor_to_data.end())
  {
    delete[] it->second;
  }

  _tensor_to_data[tensor] = data;
  return Ok;
}

Status GradientCalculationStorage::saveGradients(const circle::Tensor *tensor, uint8_t *data)
{
  const auto it = _gradients.find(tensor);
  if (it != _gradients.end())
    return Error;

  _gradients[tensor] = data;
  return Ok;
}

Status GradientCalculationStorage::getDataByTensor(const circle::Tensor *tensor, uint8_t **data)
{
  assert(tensor != nullptr); // CALLER SIDE

  auto it = _tensor_to_data.find(tensor);

  assert(it != _tensor_to_data.end() && "No data");
  if (it == _tensor_to_data.end())
  {
    return Error;
  }

  *data = it->second;

  return Ok;
}

Status GradientCalculationStorage::createGradientMatrix(const circle::Tensor *tensor)
{
  const auto rows = Tensor::dim(tensor, 0);
  const auto cols = Tensor::dim(tensor, 1);

  uint8_t *gradient_values = new uint8_t[rows * cols * size(Tensor::element_type(tensor))];

  switch (Tensor::element_type(tensor))
  {
    case DataType::FLOAT32:
    {
      float *gradient_values_float = reinterpret_cast<float *>(gradient_values);
      for (int row = 0; row < rows; ++row)
      {
        for (int col = 0; col < cols; ++col)
        {
          gradient_values_float[col + row * cols] = 0;
        }
      }
      break;
    }
    default:
    {
      assert(false && "Unsupported type");
      return Error;
    }
  }
  saveGradients(tensor, gradient_values);

  return Ok;
}

Status GradientCalculationStorage::getGradients(const circle::Tensor *tensor, uint8_t **data)
{
  assert(tensor != nullptr); // CALLER SIDE

  auto it = _gradients.find(tensor);

  if (it == _gradients.end())
  {
    Status status = createGradientMatrix(tensor);
    if (status != Ok)
      return status;
  }

  it = _gradients.find(tensor);

  *data = it->second;

  if (*data == nullptr)
    return Error;

  return Ok;
}

GradientCalculationStorage::~GradientCalculationStorage()
{
  clearComputedGradients();
  clearComputedData();
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
