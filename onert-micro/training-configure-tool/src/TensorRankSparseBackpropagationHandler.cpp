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

#include "TensorRankSparseBackpropagationHandler.h"
#include "SparseBackpropagationHelper.h"
#include "TrainingDriverHandler.h"

#include <unordered_set>
#include <cassert>

#define MODEL_TYPE float

#define PRINT 0

using namespace onert_micro;

namespace
{

} // namespace

OMStatus training_configure_tool::findBestSparseBackpropagationTensorsRanks(
  onert_micro::OMConfig &config, TrainData &train_data,
  const std::unordered_set<uint16_t> &selected_op_indexes,
  std::unordered_map<uint16_t, OpTrainableRank> &best_train_ranks)
{
  // Clear to find best values
  best_train_ranks.clear();

  // 1 - Find All combinations with ranks for current selected op indexes
  // 2 - Run all of them to find best variant

  // 1 - Find All combinations with ranks for current selected op indexes
  std::vector<std::unordered_map<uint16_t, OpTrainableRank>> all_combinations =
    findAllTensorsRanksCombinations(selected_op_indexes, config, train_data);

#if PRINT
  printf("All combinations: op_index : rank_value; { \n");
  for (const auto &combination : all_combinations)
  {
    for (auto &p : combination)
    {
      printf("(%d : %d); ", p.first, p.second);
    }
    printf("\n");
  }
  printf("}\n");

#endif // PRINT

  // 2 - Run all of them to find best variant
  TrainResult best_train_result(train_data);
  for (const auto &combination : all_combinations)
  {
#if PRINT
    printf("Current checked combination: op_index : rank_value; { ");
    for (auto &p : combination)
    {
      printf("(%d : %d); ", p.first, p.second);
    }
    printf("}\n");
#endif

    std::vector<char> tmp_buffer;
    // Create data with current buffer information
    createResultData({combination}, tmp_buffer);
    config.training_context.training_config_info_data = tmp_buffer.data();

    TrainResult train_result(train_data);
    // Run train with this information
    runTrainProcessWithCurConfig(config, train_data, train_result);

#if PRINT
    printf("Find the following result:\n");
    if (train_result.best_metrics_results.first == CROSS_ENTROPY_METRICS)
    {
      printf("CROSS_ENTROPY_METRIC = %f\n", train_result.best_metrics_results.second);
      printf("PEAK_MEMORY_RESULT = %zu\n", train_result.peak_memory_footprint);
    }
#endif

    // Compare with best result to find best
    bool cmp_result = cmpTrainResults(train_result, best_train_result, train_data.acceptable_diff);
    if (cmp_result)
    {
      // Cur rest is better
#if PRINT
      printf("BETTER RESULT\n");
#endif
      best_train_result = train_result;
      best_train_ranks = combination;
    }
  }

#if PRINT
  printf("FINISH\n");

  printf("Best rank combination: op_index : rank_value; { ");
  for (auto &p : best_train_ranks)
  {
    printf("(%d : %d); ", p.first, p.second);
  }
  printf("}\n");

  printf("Find the following result:\n");
  if (best_train_result.best_metrics_results.first == CROSS_ENTROPY_METRICS)
  {
    printf("CROSS_ENTROPY_METRIC = %f\n", best_train_result.best_metrics_results.second);
    printf("PEAK_MEMORY_RESULT = %zu\n", best_train_result.peak_memory_footprint);
  }
#endif

  return Ok;
}
