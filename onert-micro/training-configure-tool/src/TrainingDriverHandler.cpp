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

#include "TrainingDriverHandler.h"
#include "OMTrainingInterpreter.h"
#include "TrainingConfigureFileHandler.h"

#include <iostream>
#include <numeric>

using namespace onert_micro;

namespace
{

#define MODEL_TYPE float
#define PRINT 0

float findAverage(const std::vector<float> &values)
{
  auto res = std::accumulate(values.begin(), values.end(), 0.f);
  return res / static_cast<float>(values.size());
}

} // namespace

OMStatus training_configure_tool::runTrainProcessWithCurConfig(
  OMConfig &config, const training_configure_tool::TrainData &train_data, TrainResult &train_result)
{
  // Clear previous results
  train_result.peak_memory_footprint = 0;

  training_configure_tool::DataBuffer circle_model =
    training_configure_tool::readFile(train_data.circle_model_path);
  training_configure_tool::DataBuffer wof_data;
  // If defined wof file
  if (train_data.wof_file_path != nullptr)
    wof_data = training_configure_tool::readFile(train_data.wof_file_path);

  // Save model size and model ptr in config
  config.model_size = circle_model.size();
  config.model_ptr = circle_model.data();

  // If defined wof file
  if (train_data.wof_file_path != nullptr)
    config.wof_ptr = nullptr;

  config.train_mode = true;

  // Create training interpreter and import models
  onert_micro::OMTrainingInterpreter train_interpreter;
  train_interpreter.importTrainModel(config.model_ptr, config);

  const auto batch_size = config.training_context.batch_size;
  // TODO: support more inputs
  const auto input_size = train_interpreter.getInputSizeAt(0);
  const auto output_size = train_interpreter.getOutputSizeAt(0);

  // Temporary buffer to read input data from file using BATCH_SIZE
  float training_input[batch_size * input_size];
  float training_target[batch_size * output_size];
  // Note: here test size used with BATCH_SIZE = 1
  float test_input[input_size];
  float test_target[output_size];

  // Best results
  float max_accuracy = std::numeric_limits<float>::min();
  float min_mse = std::numeric_limits<float>::max();
  float min_mae = std::numeric_limits<float>::max();
  float min_entropy = std::numeric_limits<float>::max();

  const auto training_epochs = config.training_context.epochs;
  for (uint32_t e = 0; e < training_epochs; ++e)
  {
#if PRINT
    printf("Epoch: %d/%d\n", e + 1, training_epochs);
#endif
    std::vector<float> accuracy_v;
    std::vector<float> cross_entropy_v;
    std::vector<float> mse_v;
    std::vector<float> mae_v;

    // Run train for current epoch
    config.training_context.num_epoch = e + 1;
    uint32_t num_steps = train_data.num_train_data_samples / batch_size;
    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size =
        std::min(batch_size, train_data.num_train_data_samples - batch_size * i - 1);
      cur_batch_size = std::max(1u, cur_batch_size);

      config.training_context.batch_size = cur_batch_size;

      // Read current input and target data
      training_configure_tool::readDataFromFile(train_data.input_input_train_data_path,
                                                reinterpret_cast<char *>(training_input),
                                                sizeof(float) * input_size * cur_batch_size,
                                                i * sizeof(MODEL_TYPE) * input_size * batch_size);

      training_configure_tool::readDataFromFile(train_data.input_target_train_data_path,
                                                reinterpret_cast<char *>(training_target),
                                                sizeof(float) * output_size * cur_batch_size,
                                                i * sizeof(MODEL_TYPE) * output_size * batch_size);

      // Set input and target
      train_interpreter.setInput(reinterpret_cast<uint8_t *>(training_input), 0);
      train_interpreter.setTarget(reinterpret_cast<uint8_t *>(training_target), 0);

      // Train with current batch size
      train_interpreter.trainSingleStep(config);
    }

    train_interpreter.reset();

    // Reset num step value
    config.training_context.num_step = 0;
    num_steps = train_data.num_test_data_samples;

    accuracy_v.clear();
    cross_entropy_v.clear();
    mae_v.clear();
    mse_v.clear();

    if (train_data.metrics_to_check_best_config == NONE)
      continue;

    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size = 1;
      training_configure_tool::readDataFromFile(
        train_data.input_input_test_data_path, reinterpret_cast<char *>(test_input),
        sizeof(float) * input_size * cur_batch_size, i * sizeof(MODEL_TYPE) * input_size);

      training_configure_tool::readDataFromFile(
        train_data.input_target_test_data_path, reinterpret_cast<char *>(test_target),
        sizeof(float) * output_size * cur_batch_size, i * sizeof(MODEL_TYPE) * output_size);

      train_interpreter.setInput(reinterpret_cast<uint8_t *>(test_input), 0);
      train_interpreter.setTarget(reinterpret_cast<uint8_t *>(test_target), 0);

      switch (train_data.metrics_to_check_best_config)
      {
        case onert_micro::CROSS_ENTROPY_METRICS:
        {
          float cross_entropy_metric = 0.f;
          train_interpreter.evaluateMetric(config, onert_micro::CROSS_ENTROPY_METRICS,
                                           reinterpret_cast<void *>(&cross_entropy_metric),
                                           cur_batch_size);
          cross_entropy_v.push_back(cross_entropy_metric);
        }
        break;
        case onert_micro::ACCURACY:
        {
          float accuracy = 0.f;
          train_interpreter.evaluateMetric(config, onert_micro::ACCURACY,
                                           reinterpret_cast<void *>(&accuracy), cur_batch_size);
          accuracy_v.push_back(accuracy);
        }
        break;
        case onert_micro::MSE_METRICS:
        {
          float mse = 0.f;
          train_interpreter.evaluateMetric(config, onert_micro::MSE_METRICS,
                                           reinterpret_cast<void *>(&mse), cur_batch_size);
          mse_v.push_back(mse);
        }
        break;
        case onert_micro::MAE_METRICS:
        {
          float mae = 0.f;
          train_interpreter.evaluateMetric(config, onert_micro::MAE_METRICS,
                                           reinterpret_cast<void *>(&mae), cur_batch_size);
          mae_v.push_back(mae);
        }
        break;
        default:
        {
          assert(false && "Not supported");
          return UnsupportedType;
        }
      }
    }

    // Calculate and use average values
    switch (train_data.metrics_to_check_best_config)
    {
      case onert_micro::CROSS_ENTROPY_METRICS:
      {
        auto average_value = findAverage(cross_entropy_v);
        if (average_value < min_entropy)
          min_entropy = average_value;
      }
      break;
      case onert_micro::ACCURACY:
      {
        auto average_value = findAverage(accuracy_v);
        if (average_value > max_accuracy)
          max_accuracy = average_value;
      }
      break;
      case onert_micro::MSE_METRICS:
      {
        auto average_value = findAverage(mse_v);
        if (average_value < min_mse)
          min_mse = average_value;
      }
      break;
      case onert_micro::MAE_METRICS:
      {
        auto average_value = findAverage(mae_v);
        if (average_value < min_mae)
          min_mae = average_value;
      }
      break;
      default:
      {
        assert(false && "Not supported");
        return UnsupportedType;
      }
    }
  }
  train_result.peak_memory_footprint = train_interpreter.getPeakFootprintMemory();
  switch (train_data.metrics_to_check_best_config)
  {
    case onert_micro::CROSS_ENTROPY_METRICS:
    {
      train_result.best_metrics_results = {train_data.metrics_to_check_best_config, min_entropy};
    }
    break;
    case onert_micro::ACCURACY:
    {
      train_result.best_metrics_results = {train_data.metrics_to_check_best_config, max_accuracy};
    }
    break;
    case onert_micro::MSE_METRICS:
    {
      train_result.best_metrics_results = {train_data.metrics_to_check_best_config, min_mse};
    }
    break;
    case onert_micro::MAE_METRICS:
    {
      train_result.best_metrics_results = {train_data.metrics_to_check_best_config, min_mae};
    }
    break;
    case onert_micro::NONE:
    {
      break;
    }
    default:
    {
      assert(false && "Not supported");
      return UnsupportedType;
    }
  }
  return Ok;
}
