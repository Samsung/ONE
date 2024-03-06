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
  float _lambda = 0.0f;
  uint16_t _batches = 0;
  std::vector<uint16_t> _targets_indexes;

public:
  OMStatus initTrainingStorage(core::reader::OMCircleReader *reader, const OMConfig &config);

  // Getters
  float getLambda() { return _lambda; }
  uint16_t getBatches() { return _batches; }
  std::vector<uint16_t> &getTargetsIndexes() { return _targets_indexes; }
  std::unordered_map<uint16_t, uint16_t> &getBackpropIndexesToMainIndexesTable() { return _backprop_indexes_to_main_indexes_table; }
};

} // namespace train
} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_TRAIN_TRAINING_STORAGE_H
