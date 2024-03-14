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

#ifndef ONERT_MICRO_CORE_TRAIN_TRAINING_STORAGE_H
#define ONERT_MICRO_CORE_TRAIN_TRAINING_STORAGE_H

#include "OMStatus.h"
#include "OMConfig.h"

#include "core/reader/OMCircleReader.h"

#include <vector>
#include <unordered_map>
#include <cstdint>

namespace onert_micro
{
namespace core
{
namespace train
{

// Struct to save training features
class OMTrainingStorage
{
private:
  std::unordered_map<uint16_t, uint16_t> _backprop_indexes_to_main_indexes_table = {};
  std::vector<uint16_t> _targets_indexes;
  std::unordered_map<uint16_t, uint8_t *> _tensor_to_exponent_avg_squares = {};
  std::unordered_map<uint16_t, uint8_t *> _tensor_to_exponent_avg = {};
  std::unordered_map<uint16_t, uint8_t *> _gradients_storage = {};
  uint16_t _adam_step = 0;
  onert_micro::OMTrainingConfig _training_configs;

public:
  OMTrainingStorage() = default;
  OMTrainingStorage(const OMTrainingStorage &) = delete;
  OMTrainingStorage &operator=(const OMTrainingStorage &) = delete;
  OMTrainingStorage &&operator=(const OMTrainingStorage &&) = delete;
  OMTrainingStorage(OMTrainingStorage &&) = default;

  ~OMTrainingStorage();

  OMStatus initTrainingStorage(core::reader::OMCircleReader *reader, const OMConfig &config);

  // Getters
  float getLambda() { return _training_configs.lambda; }
  float getBeta() { return _training_configs.beta; }
  float getBetaSquares() { return _training_configs.beta_squares; }
  float getEpsilon() { return _training_configs.epsilon; }
  uint16_t &getAdamStep() { return _adam_step; }
  uint16_t &getBatches() { return _training_configs.batches; }

  // Exponent average squares
  uint8_t *getExponentAvgSquaresData(uint16_t tensor_index);
  uint8_t *getExponentAvgData(uint16_t tensor_index);
  uint8_t *getGradientData(uint16_t tensor_index);

  void reset();

  std::vector<uint16_t> &getTargetsIndexes() { return _targets_indexes; }
  std::unordered_map<uint16_t, uint16_t> &getBackpropIndexesToMainIndexesTable() { return _backprop_indexes_to_main_indexes_table; }

  onert_micro::OMOptimizationStrategy getOptimizationStrategy() { return _training_configs.optimization_strategy; }
};

} // namespace train
} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_TRAIN_TRAINING_STORAGE_H
