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

#include "SparseBackpropagationHandler.h"
#include "SparseBackpropagationHelper.h"
#include "TrainingDriverHandler.h"

#include <unordered_set>
#include <cassert>

#define MODEL_TYPE float

#define PRINT 0

using namespace onert_micro;

OMStatus training_configure_tool::findBestTrainableOpIndexes(
  OMConfig &config, training_configure_tool::TrainData &train_data,
  std::unordered_set<uint16_t> &best_trainable_op_indexes)
{
  // Clear to find best values
  best_trainable_op_indexes.clear();
  // 1 - Find all trainable ops indexes in the model - initial_train_op_indexes
  // 2 - Generate all possible sets from initial_train_op_indexes
  // 3 - If above memory restriction is defined - then remove operations indexes sets with peak
  // memory footprint greater then given restriction 4 - Try all found sets to find best metrics
  // results

  // 1 - Find all trainable ops indexes in the model - initial_train_op_indexes
  std::unordered_set<uint16_t> initial_train_op_indexes =
    training_configure_tool::findAllTrainableOps(train_data.circle_model_path);
  assert(!initial_train_op_indexes.empty());
  if (initial_train_op_indexes.empty())
    return UnknownError;
#if PRINT
  printf("Found next trainable indexes in the model: ");
  for (auto i : initial_train_op_indexes)
  {
    printf("%d ", i);
  }
  printf("\n");
#endif

  // 2 - Generate all possible sets from initial_train_op_indexes
  std::vector<std::unordered_set<uint16_t>> all_possible_train_op_indexes_sets =
    training_configure_tool::generateAllPossibleOpIndexesSets(initial_train_op_indexes);
  assert(all_possible_train_op_indexes_sets.empty() == false);
  if (all_possible_train_op_indexes_sets.empty() == true)
    return UnknownError;
#if PRINT
  printf("Found %zu unique trainable ops indexes in the model:\n",
         all_possible_train_op_indexes_sets.size());
  for (const auto &s : all_possible_train_op_indexes_sets)
  {
    printf("Op indexes set = { ");
    for (auto i : s)
    {
      printf("%d ", i);
    }
    printf("}\n");
  }
#endif
  // Clear initial due to is not needed
  initial_train_op_indexes.clear();

  // 3 - If above memory restriction is defined, then save only sets with peak memory less then
  // restriction
  std::vector<std::unordered_set<uint16_t>> selected_op_indexes_sets =
    training_configure_tool::selectOpIndexesSetsAccordingToMemoryRestriction(
      all_possible_train_op_indexes_sets, config, train_data);
#if PRINT
  printf("Found %zu selected op indexes sets:\n", selected_op_indexes_sets.size());
  for (const auto &s : selected_op_indexes_sets)
  {
    printf("Op indexes set = { ");
    for (auto i : s)
    {
      printf("%d ", i);
    }
    printf("}\n");
  }
#endif
  // Clear not needed object
  all_possible_train_op_indexes_sets.clear();

  // 4 - Try all found sets to find best metrics results
  // To save best values
  TrainResult best_train_result(train_data);
  for (const auto &index_set : selected_op_indexes_sets)
  {
#if PRINT
    printf("Current checked op indexes set = { ");
    for (auto i : index_set)
    {
      printf("%d ", i);
    }
    printf("}\n");
#endif

    // Construct mapping with current indexes - use default train ALL parts
    std::unordered_map<uint16_t, OpTrainableRank> train_op_ranks;
    for (auto index : index_set)
    {
      train_op_ranks[index] = ALL;
    }

    std::vector<char> tmp_buffer;
    // Create data with current buffer information
    createResultData({train_op_ranks}, tmp_buffer);
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
      best_trainable_op_indexes = index_set;
    }
  }

#if PRINT
  printf("FINISH\n");

  printf("Best op indexes set = { ");
  for (auto i : best_trainable_op_indexes)
  {
    printf("%d ", i);
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
