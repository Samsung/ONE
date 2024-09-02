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

#include "SparseBackpropagationHelper.h"
#include "TrainingConfigureFileHandler.h"
#include "TrainingDriverHandler.h"
#include "TrainingConfigureFileHandler.h"
#include "core/reader/OMCircleReader.h"

#include <cassert>

#define MODEL_TYPE float
#define PRINT 0

using namespace onert_micro;

namespace
{

bool isTrainableWeights(const circle::OperatorCode *opcode)
{
  switch (opcode->builtin_code())
  {
    case circle::BuiltinOperator_FULLY_CONNECTED:
    case circle::BuiltinOperator_CONV_2D:
      return true;
    default:
      return false;
  }
}

void generateRankCombinations(
  const std::unordered_map<uint16_t, std::unordered_set<uint16_t>> &index_to_possible_ranks,
  const std::vector<uint16_t> &indices, size_t currentIndex,
  std::vector<uint16_t> &currentCombination, std::vector<std::vector<uint16_t>> &result)
{
  if (currentIndex == indices.size())
  {
    result.push_back(currentCombination);
    return;
  }
  uint16_t index = indices[currentIndex];
  for (uint16_t rank : index_to_possible_ranks.at(index))
  {
    currentCombination.push_back(rank);
    generateRankCombinations(index_to_possible_ranks, indices, currentIndex + 1, currentCombination,
                             result);
    currentCombination.pop_back();
  }
}

// Find pairs: selected ops indexes - divided dim max rank value
std::unordered_map<uint16_t, uint32_t> findTrainableTensorsMaxDivideRankAccordingToOperatorIndex(
  const std::unordered_set<uint16_t> &selected_op_indexes,
  const onert_micro::core::reader::OMCircleReader &reader)
{
  std::unordered_map<uint16_t, uint32_t> operator_index_to_tensor_max_divide_rank;

  // Read ops
  auto operators = reader.operators();
  assert(operators != nullptr);

  auto op_size = operators->size();

  // Obtain operation codes
  auto op_codes = reader.opcodes();

  const auto tensors = reader.tensors();

  // Go over selected best op indexes
  for (auto op_index : selected_op_indexes)
  {
    auto cur_op = operators->operator[](op_index);

    // Get opcode index
    uint32_t cur_opcode_index = cur_op->opcode_index();
    assert(cur_opcode_index < op_codes->size());

    const auto opcode = op_codes->operator[](cur_opcode_index);

    const auto inputs_tensors = cur_op->inputs();

    uint32_t tensor_divided_dim_value = 0;

    switch (opcode->builtin_code())
    {
      case circle::BuiltinOperator_FULLY_CONNECTED:
      case circle::BuiltinOperator_CONV_2D:
      {
        assert(inputs_tensors->size() >= 2);
        auto tensor_index = inputs_tensors->operator[](1);
        assert(tensor_index != -1);
        assert(tensor_index < tensors->size());
        auto tensor = tensors->operator[](tensor_index);

        // For FC and Conv2D op tool provide rank over 0 dimensional
        tensor_divided_dim_value = tensor->shape()->operator[](0);

        break;
      }
      default:
        assert(false && "Unsupported type");
        tensor_divided_dim_value = 0; // Not supported
    }

    assert(tensor_divided_dim_value != 0);
    operator_index_to_tensor_max_divide_rank[op_index] = tensor_divided_dim_value;
  }

  return operator_index_to_tensor_max_divide_rank;
}

void recursiveGenerateAllPossibleOpIndexesSetsHelper(
  std::vector<std::unordered_set<uint16_t>> &result, std::unordered_set<uint16_t> &cur_set,
  std::unordered_set<uint16_t>::const_iterator cur_it_set_value,
  std::unordered_set<uint16_t>::const_iterator &end_it_set)
{
  // If we reach end of the initial set then finish
  if (cur_it_set_value == end_it_set)
  {
    // If set is not empty add to final result
    if (cur_set.empty() == false)
      result.push_back(cur_set);
    return;
  }

  // Add value to current set
  uint16_t cur_index = *cur_it_set_value;
  cur_set.insert(cur_index);
  // Run further and move iterator to next position
  cur_it_set_value++;
  recursiveGenerateAllPossibleOpIndexesSetsHelper(result, cur_set, cur_it_set_value, end_it_set);
  // Remove current index from set
  cur_set.erase(cur_index);
  // Run again recursive functions but now without current index
  recursiveGenerateAllPossibleOpIndexesSetsHelper(result, cur_set, cur_it_set_value, end_it_set);
}

} // namespace

// Find is left train result is better then right
bool training_configure_tool::cmpTrainResults(const training_configure_tool::TrainResult &left,
                                              const training_configure_tool::TrainResult &right,
                                              const float acceptable_diff)
{
  // Metrics should be the same
  assert(left.best_metrics_results.first == right.best_metrics_results.first);
  OMMetrics metric = left.best_metrics_results.first;
  float left_metric_res = left.best_metrics_results.second;
  float right_metric_res = right.best_metrics_results.second;

  bool is_in_acceptable_diff = std::abs(left_metric_res - right_metric_res) <= acceptable_diff;
  if (is_in_acceptable_diff)
  {
    return left.peak_memory_footprint < right.peak_memory_footprint;
  }

  switch (metric)
  {
    case onert_micro::ACCURACY:
    {
      return left.best_metrics_results.second > right.best_metrics_results.second;
    }
    break;
    case onert_micro::CROSS_ENTROPY_METRICS:
    case onert_micro::MSE_METRICS:
    case onert_micro::MAE_METRICS:
    {

      return left.best_metrics_results.second < right.best_metrics_results.second;
    }
    break;
    default:
      assert(false && "Unsupported type");
      break;
  }
  return true;
}

// Remove operations indexes sets with peak memory footprint greater then given restriction:
//    1 - Run train interpreter with all this sets with single train sample and single test sample
//    to obtain approximately peak memory footprint for each set 2 - Cut according to max peak
//    memory
std::vector<std::unordered_set<uint16_t>>
training_configure_tool::selectOpIndexesSetsAccordingToMemoryRestriction(
  const std::vector<std::unordered_set<uint16_t>> &op_indexes_sets, onert_micro::OMConfig config,
  training_configure_tool::TrainData train_data)
{
  // It 0 - then is not set
  if (train_data.memory_above_restriction == 0)
  {
    return op_indexes_sets;
  }

  std::vector<std::unordered_set<uint16_t>> result;

  // To obtain real estimation we need minimum batch_size = 2 and num_train_data_samples = 4
  // Change config train and test sample values
  train_data.num_test_data_samples = 0;
  train_data.num_train_data_samples = std::min(4, train_data.num_train_data_samples);
  // To disable tests
  train_data.metrics_to_check_best_config = NONE;
  // Set number of the epochs and batch size to one
  config.training_context.epochs = 1;
  config.training_context.batch_size = std::min(2u, config.training_context.batch_size);

  for (const auto &op_indexes_set : op_indexes_sets)
  {
#if PRINT
    printf("Start checking: { ");
    for (auto i : op_indexes_set)
    {
      printf("%d ", i);
    }
    printf("}\n");
#endif
    // Construct mapping with current indexes - use default train ALL parts
    std::unordered_map<uint16_t, OpTrainableRank> train_op_ranks;
    for (auto index : op_indexes_set)
    {
      train_op_ranks[index] = ALL;
    }

    std::vector<char> tmp_buffer;
    // Create data with current buffer information
    createResultData({train_op_ranks}, tmp_buffer);
    config.training_context.training_config_info_data = tmp_buffer.data();

    TrainResult train_result;
    // Run train with this information
    runTrainProcessWithCurConfig(config, train_data, train_result);
#if PRINT
    printf("CURRENT MEMORY PEAK = %zu\n", train_result.peak_memory_footprint);
#endif
    if (train_result.peak_memory_footprint < train_data.memory_above_restriction)
    {
#if PRINT
      printf("Added to the result\n");
#endif
      result.push_back(op_indexes_set);
    }
  }

  return result;
}

// To generate all possible sets from initial_train_op_indexes
std::vector<std::unordered_set<uint16_t>> training_configure_tool::generateAllPossibleOpIndexesSets(
  const std::unordered_set<uint16_t> &initial_train_op_indexes)
{
  std::vector<std::unordered_set<uint16_t>> result;
  std::unordered_set<uint16_t> cur_set;

  auto begin_it = initial_train_op_indexes.begin();
  auto end_it = initial_train_op_indexes.end();
  recursiveGenerateAllPossibleOpIndexesSetsHelper(result, cur_set, begin_it, end_it);

  return result;
}

// To find all trainable ops indexes in the model - initial_train_op_indexes
std::unordered_set<uint16_t>
training_configure_tool::findAllTrainableOps(const char *circle_model_path)
{
  std::unordered_set<uint16_t> result;

  training_configure_tool::DataBuffer model_ptr =
    training_configure_tool::readFile(circle_model_path);

  // Init reader
  OMStatus status = Ok;
  core::reader::OMCircleReader reader;
  assert(model_ptr.data() != nullptr);
  status = reader.parse(model_ptr.data());
  assert(status == Ok);
  // return empty set
  if (status != Ok)
    return result;
  // TODO: support multi subgraph models
  status = reader.select_subgraph(0);
  // return empty set
  if (status != Ok)
    return result;

  // Read ops
  auto operators = reader.operators();
  assert(operators != nullptr);

  auto op_size = operators->size();

  // Obtain operation codes
  auto op_codes = reader.opcodes();

  // Run through all ops
  for (uint32_t i = 0; i < op_size; ++i)
  {
    auto cur_op = operators->operator[](i);

    // Get opcode index
    uint32_t cur_opcode_index = cur_op->opcode_index();
    assert(cur_opcode_index < op_codes->size());

    const auto opcode = op_codes->operator[](cur_opcode_index);

    // If op is trainable - insert it
    if (isTrainableWeights(opcode))
      result.insert(static_cast<uint16_t>(i));
  }

  return result;
}

std::vector<std::unordered_map<uint16_t, training_configure_tool::OpTrainableRank>>
training_configure_tool::findAllTensorsRanksCombinations(
  const std::unordered_set<uint16_t> &selected_op_indexes, onert_micro::OMConfig config,
  training_configure_tool::TrainData train_data)
{
  // 1 - Find pairs: selected ops indexes - divided dim max rank value
  // 2 - Find for every tensor index every possible rank according to its opcode and size
  // 3 - Get result
  std::vector<std::unordered_map<uint16_t, training_configure_tool::OpTrainableRank>> result;

  training_configure_tool::DataBuffer model_ptr =
    training_configure_tool::readFile(train_data.circle_model_path);

  // Init reader
  OMStatus status = Ok;
  core::reader::OMCircleReader reader;
  assert(model_ptr.data() != nullptr);
  status = reader.parse(model_ptr.data());
  assert(status == Ok);
  // return empty set
  if (status != Ok)
    return result;
  // TODO: support multi subgraph models
  status = reader.select_subgraph(0);
  // return empty set
  if (status != Ok)
    return result;

  // 1 - Find pairs: selected ops indexes - divided dim max rank value
  std::unordered_map<uint16_t, uint32_t> operator_index_to_tensor_index =
    findTrainableTensorsMaxDivideRankAccordingToOperatorIndex(selected_op_indexes, reader);
  assert(operator_index_to_tensor_index.size() == selected_op_indexes.size());
  // 2 - Find for every tensor index every possible rank according to its opcode and size
  std::unordered_map<uint16_t, std::unordered_set<uint16_t>> op_index_to_all_possible_ranks;
  for (auto &p : operator_index_to_tensor_index)
  {
    const auto op_index = p.first;
    const auto max_value = p.second;

    uint16_t cur_value = 2;
    op_index_to_all_possible_ranks[op_index] = {ALL, ONLY_BIAS};
    while (cur_value < uint32_t(OpTrainableRank::MAX_VALUE) and cur_value <= max_value)
    {
      auto new_value = cur_value * 2;
      while (cur_value < uint16_t(OpTrainableRank::MAX_VALUE) and cur_value < new_value)
      {
        op_index_to_all_possible_ranks[op_index].insert(cur_value);
        cur_value++;
      }
    }
  }
  // Get all op indices
  std::vector<uint16_t> indices(selected_op_indexes.begin(), selected_op_indexes.end());
  std::vector<std::vector<uint16_t>> rank_combinations;
  std::vector<uint16_t> cur_v;
  generateRankCombinations(op_index_to_all_possible_ranks, indices, 0, cur_v, rank_combinations);

  for (const auto &ranks : rank_combinations)
  {
    std::unordered_map<uint16_t, OpTrainableRank> combination;
    for (size_t i = 0; i < indices.size(); ++i)
    {
      combination[indices[i]] = OpTrainableRank(ranks[i]);
    }
    result.push_back(std::move(combination));
  }

  return result;
}
