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

#include "luci_interpreter/TrainingOnertMicro.h"
#include "core/TrainingModule.h"

#include <cmath>

namespace luci_interpreter
{

namespace training
{

namespace
{

float calculateMSEError(const float *predicted_values, const float *target_values,
                        const uint32_t output_size)
{
  const uint32_t output_number_values = output_size / sizeof(float);
  float result = 0.0f;
  for (int i = 0; i < output_number_values; ++i)
  {
    result += std::pow(predicted_values[i] - target_values[i], 2);
  }

  return result / output_number_values;
}

float calculateMAEError(const float *predicted_values, const float *target_values,
                        const uint32_t output_size)
{
  const uint32_t output_number_values = output_size / sizeof(float);
  float result = 0.0f;
  for (int i = 0; i < output_number_values; ++i)
  {
    result += std::abs(predicted_values[i] - target_values[i]);
  }

  return result / output_number_values;
}

Status calculateError(const uint8_t *predicted_value, const uint8_t *target_value, void *result,
                      const uint32_t output_size, MetricsTypeEnum error_type)
{
  switch (error_type)
  {
    case MSE:
    {
      float *result_float = reinterpret_cast<float *>(result);
      *result_float +=
        calculateMSEError(reinterpret_cast<const float *>(predicted_value),
                          reinterpret_cast<const float *>(target_value), output_size);
      break;
    }
    case MAE:
    {
      float *result_float = reinterpret_cast<float *>(result);
      *result_float +=
        calculateMAEError(reinterpret_cast<const float *>(predicted_value),
                          reinterpret_cast<const float *>(target_value), output_size);
      break;
    }
    default:
    {
      return Error;
    }
  }

  return Ok;
}

} // namespace

TrainingOnertMicro::TrainingOnertMicro(Interpreter *interpreter, TrainingSettings &settings)
  : _interpreter(interpreter), _settings(settings), _is_training_mode(false),
    _module(&interpreter->_runtime_module)
{
  // Do nothing
}

Status TrainingOnertMicro::enableTrainingMode()
{
  if (_is_training_mode)
  {
    return DoubleTrainModeError;
  }

  const Status status = _module.enableTrainingMode(_settings, &_interpreter->_memory_manager);

  if (status != Ok)
    assert("Some error during enabling training mode");

  _is_training_mode = true;

  return status;
}

Status TrainingOnertMicro::disableTrainingMode(bool resetWeights)
{
  if (_is_training_mode == false)
  {
    return Ok;
  }

  const Status status = _module.disableTrainingMode(resetWeights);

  if (status != Ok)
    assert("Some error during disabling training mode");

  _is_training_mode = false;

  return status;
}

Status TrainingOnertMicro::train(uint32_t number_of_train_samples, const uint8_t *train_data,
                                 const uint8_t *label_train_data)
{
  if (_is_training_mode == false)
    return EnableTrainModeError;

  const uint32_t batch_size = _settings.batch_size;

  const uint32_t num_inferences = number_of_train_samples / batch_size;

  const uint32_t remains = number_of_train_samples % batch_size;

  const uint32_t epochs = _settings.number_of_epochs;

  const int32_t input_tensor_size = _interpreter->getInputDataSizeByIndex(0);
  const int32_t output_tensor_size = _interpreter->getOutputDataSizeByIndex(0);

  const uint8_t *cur_train_data = train_data;
  const uint8_t *cur_label_train_data = label_train_data;

  for (uint32_t epoch = 0; epoch < epochs; ++epoch)
  {
    for (uint32_t infer = 0; infer < num_inferences; ++infer)
    {
      for (uint32_t batch = 0; batch < batch_size; ++batch)
      {
        _interpreter->allocateAndWriteInputTensor(0, cur_train_data, input_tensor_size);

        _interpreter->interpret();

        _module.computeGradients(_settings, cur_label_train_data);
        cur_train_data += input_tensor_size;
        cur_label_train_data += output_tensor_size;
      }

      _module.updateWeights(_settings);
    }
    cur_train_data = train_data;
    cur_label_train_data = label_train_data;
  }

  return Ok;
}

Status TrainingOnertMicro::test(uint32_t number_of_train_samples, const uint8_t *test_data,
                                const uint8_t *label_test_data, void *metric_value_result)
{
  const int32_t input_tensor_size = _interpreter->getInputDataSizeByIndex(0);
  const int32_t output_tensor_size = _interpreter->getOutputDataSizeByIndex(0);

  const uint8_t *cur_test_data = test_data;
  const uint8_t *cur_label_test_data = label_test_data;

  switch (_settings.metric)
  {
    case MSE:
    case MAE:
    {
      float *result_float = reinterpret_cast<float *>(metric_value_result);
      *result_float = 0.0f;
      break;
    }
    default:
    {
      return Error;
    }
  }

  for (uint32_t sample = 0; sample < number_of_train_samples; ++sample)
  {
    _interpreter->allocateAndWriteInputTensor(0, cur_test_data, input_tensor_size);

    _interpreter->interpret();

    const uint8_t *output_data = _interpreter->readOutputTensor(0);

    Status status = calculateError(output_data, cur_label_test_data, metric_value_result,
                                   output_tensor_size, _settings.metric);

    if (status != Ok)
      return status;

    cur_test_data += input_tensor_size;
    cur_label_test_data += output_tensor_size;
  }

  switch (_settings.metric)
  {
    case MSE:
    case MAE:
    {
      float *result_float = reinterpret_cast<float *>(metric_value_result);
      *result_float /= number_of_train_samples;
      break;
    }
    default:
    {
      return Error;
    }
  }

  return Ok;
}

TrainingOnertMicro::~TrainingOnertMicro() { disableTrainingMode(); }

} // namespace training

} // namespace luci_interpreter

#endif // ENABLE_TRAINING
