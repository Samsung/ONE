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

#ifndef ONERT_MICRO_TRAINING_CONFIG_TOOL_SPARSE_BACKPROPAGATION_HELPER
#define ONERT_MICRO_TRAINING_CONFIG_TOOL_SPARSE_BACKPROPAGATION_HELPER

#include "OMStatus.h"
#include "OMConfig.h"
#include "TrainConfigData.h"

#include <vector>
#include <unordered_set>

namespace training_configure_tool
{

// Find is left train result is better then right in terms of metric result and memory consumptions.
// acceptable_diff - acceptable difference in metric values in order to select the best result in
// memory.
bool cmpTrainResults(const training_configure_tool::TrainResult &left,
                     const training_configure_tool::TrainResult &right,
                     const float acceptable_diff);

// To find all trainable ops indexes in the model - initial_train_op_indexes
std::unordered_set<uint16_t> findAllTrainableOps(const char *circle_model_path);

// To generate all possible sets from initial_train_op_indexes
std::vector<std::unordered_set<uint16_t>>
generateAllPossibleOpIndexesSets(const std::unordered_set<uint16_t> &initial_train_op_indexes);

// Remove operations indexes sets with peak memory footprint greater then given restriction:
//    1 - Run train interpreter with all this sets with single train sample and single test sample
//    to obtain approximately peak memory footprint for each set.
//    2 - Cut according to max peak memory.
std::vector<std::unordered_set<uint16_t>> selectOpIndexesSetsAccordingToMemoryRestriction(
  const std::vector<std::unordered_set<uint16_t>> &op_indexes_sets, onert_micro::OMConfig config,
  training_configure_tool::TrainData train_data);

// Find All combinations with ranks for current selected op indexes.
// Return vector of all possible combinations of train rank for every op.
std::vector<std::unordered_map<uint16_t, OpTrainableRank>>
findAllTensorsRanksCombinations(const std::unordered_set<uint16_t> &selected_op_indexes,
                                onert_micro::OMConfig config,
                                training_configure_tool::TrainData train_data);

} // namespace training_configure_tool

#endif // ONERT_MICRO_TRAINING_CONFIG_TOOL_SPARSE_BACKPROPAGATION_HELPER
