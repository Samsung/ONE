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

#include "Builders.h"
#include "kernels/Utils.h"

namespace luci_interpreter
{

namespace training
{

namespace
{

Status computeGradients(const circle::Operator *op, CircleReader *reader,
                        GradientCalculationStorage *gradient_calculation_storage,
                        const circle::Tensor *weight, const float *label_train_data,
                        bool is_last_layer)
{

  const auto input_index = op->inputs()->operator[](0);
  const auto output_index = op->outputs()->operator[](0);

  assert(input_index != -1);
  assert(output_index != -1);

  const auto input = reader->tensors()[input_index];
  const auto output = reader->tensors()[output_index];

  assert(input != nullptr);
  assert(output != nullptr);

  uint8_t *input_data = nullptr;
  Status status = gradient_calculation_storage->getDataByTensor(input, &input_data);

  assert(input_data != nullptr);

  float *input_data_float = reinterpret_cast<float *>(input_data);

  if (status != Ok)
    return status;

  uint8_t *output_data = nullptr;
  status = gradient_calculation_storage->getDataByTensor(output, &output_data);

  assert(input_data != nullptr);

  float *output_data_float = reinterpret_cast<float *>(output_data);

  if (status != Ok)
    return status;

  uint8_t *gradients_values = nullptr;

  const auto rows = Tensor::dim(weight, 0);
  const auto cols = Tensor::dim(weight, 1);

  status = gradient_calculation_storage->getGradients(weight, &gradients_values);
  float *gradient_values_float = reinterpret_cast<float *>(gradients_values);
  assert(gradient_values_float != nullptr);

  if (status != Ok or gradient_values_float == nullptr)
    return status;

  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      gradient_values_float[col + row * cols] = 0;
    }
  }
  if (is_last_layer)
  {
    for (int row = 0; row < rows; ++row)
    {
      for (int col = 0; col < cols; ++col)
      {
        gradient_values_float[col + row * cols] +=
          (output_data_float[row] - label_train_data[row]) * input_data_float[col];
      }
    }
  }
  else
  {
    //If not last layer - then there is gradient in label_train_data
    for (int row = 0; row < rows; ++row)
    {
      for (int col = 0; col < cols; ++col)
      {
        gradient_values_float[col + row * cols] += label_train_data[row] * input_data_float[col];
      }
    }
  }

  return Ok;
}

Status updateWeights(const circle::Operator *op, CircleReader *reader,
                     GradientCalculationStorage *gradient_calculation_storage,
                     const TrainingSettings &settings, TrainableWeightStorage *weight_storage,
                     const circle::Tensor *weight)
{
  uint8_t *weight_data = nullptr;
  Status status = weight_storage->getTrainWeightDataByTensor(weight, &weight_data);

  assert(weight_data != nullptr);

  float *weight_data_float = reinterpret_cast<float *>(weight_data);

  if (status != Ok)
    return status;

  uint8_t *gradients_values = nullptr;

  const auto rows = Tensor::dim(weight, 0);
  const auto cols = Tensor::dim(weight, 1);

  status = gradient_calculation_storage->getGradients(weight, &gradients_values);
  float *gradient_values_float = nullptr;

  gradient_values_float = reinterpret_cast<float *>(gradients_values);

  assert(gradient_values_float != nullptr);

  if (status != Ok or gradient_values_float == nullptr)
    return status;

  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      weight_data_float[col + row * cols] -=
        settings.learning_rate * gradient_values_float[col + row * cols] / settings.batch_size;
    }
  }

  return Ok;
}

} // namespace

Status train_kernel_CircleFullyConnected(const circle::Operator *op, CircleReader *reader,
                                         GradientCalculationStorage *gradient_calculation_storage,
                                         const TrainingSettings &settings,
                                         TrainableWeightStorage *weight_storage,
                                         const uint8_t *label_train_data)
{
  const auto weight_index = op->inputs()->operator[](1);
  assert(weight_index != -1);
  const auto weights = reader->tensors()[weight_index];
  assert(weights != nullptr);

  // TODO add template and switch via type
  if (label_train_data != nullptr)
    return computeGradients(op, reader, gradient_calculation_storage, weights,
                            reinterpret_cast<const float *>(label_train_data), settings.is_last_layer);

  return updateWeights(op, reader, gradient_calculation_storage, settings, weight_storage, weights);
}
} // namespace training
} // namespace luci_interpreter
