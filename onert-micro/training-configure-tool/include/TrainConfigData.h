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

#ifndef ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAIN_CONFIG_DATA
#define ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAIN_CONFIG_DATA

#include "OMConfig.h"

#include <vector>
#include <limits>
#include <cassert>
#include <unordered_set>
#include <unordered_map>

namespace training_configure_tool
{

// Enum  to indicate the degree(rank) to which part of the operation we will train:
// this is an indicator of how much data of the current operation we will train
// (for example, the entire operation, only the bias, only the upper half, and so on)
enum OpTrainableRank
{
  ALL = 0,            // 0 - Train all weights in the operation
  ONLY_BIAS = 1,      // 1 - Train bias only in the operation
  UP_1_2_PART = 2,    // 2 - Train the upper 1/2 part of the operation
  LOWER_1_2_PART = 3, // 3 - Train the lower 1/2 part of the operation
  MAX_VALUE = 4,
  // TODO add more
};

// Information for saving the data necessary for training.
// metrics_to_check_best_config - the metric by which the best configuration will be selected.
// acceptable_diff - acceptable difference in metric values in order to select the best result in
// memory. memory_above_restriction - the upper limit of memory that cannot be exceeded
struct TrainData
{
  const char *circle_model_path = nullptr;
  const char *wof_file_path = nullptr;
  const char *output_tool_file_path = nullptr;
  const char *input_input_train_data_path = nullptr;
  const char *input_target_train_data_path = nullptr;
  const char *input_input_test_data_path = nullptr;
  const char *input_target_test_data_path = nullptr;
  int32_t num_train_data_samples = 0;
  int32_t num_test_data_samples = 0;
  onert_micro::OMMetrics metrics_to_check_best_config = {};
  float acceptable_diff = 0.01;
  size_t memory_above_restriction = 0;
};

// Struct to save data which will be saved in result file
struct TrainConfigFileData
{
  std::unordered_map<uint16_t, OpTrainableRank> trainable_op_indexes_with_ranks;
};

// Information that is the result of training
// best_metrics_results - obtained best metric result during training
// peak_memory_footprint - peak memory footprint obtained during training
struct TrainResult
{
  std::pair<onert_micro::OMMetrics, float> best_metrics_results = {};
  size_t peak_memory_footprint = 0;

  TrainResult() = default;
  explicit TrainResult(TrainData train_data)
  {
    peak_memory_footprint = std::numeric_limits<size_t>::max();
    switch (train_data.metrics_to_check_best_config)
    {
      case onert_micro::ACCURACY:
        best_metrics_results = {onert_micro::ACCURACY, 0.f};
        break;
      case onert_micro::CROSS_ENTROPY_METRICS:
        best_metrics_results = {onert_micro::CROSS_ENTROPY_METRICS,
                                std::numeric_limits<float>::max()};
        break;
      case onert_micro::MSE_METRICS:
        best_metrics_results = {onert_micro::MSE_METRICS, std::numeric_limits<float>::max()};
        break;
      case onert_micro::MAE_METRICS:
        best_metrics_results = {onert_micro::MAE_METRICS, std::numeric_limits<float>::max()};
        break;
      default:
        assert(false && "Unsupported type");
        break;
    }
  }
};

} // namespace training_configure_tool

#endif // ONERT_MICRO_TRAINING_CONFIG_TOOL_TRAIN_CONFIG_DATA
