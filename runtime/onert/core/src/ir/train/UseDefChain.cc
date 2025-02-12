/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/train/UseDefChain.h"

namespace onert::ir::train
{

void UseDefChain::insertTrainingUse(const TrainingOperationIndex &idx) { _uses.insert(idx); }

void UseDefChain::removeTrainingUse(const TrainingOperationIndex &idx) { _uses.erase(idx); }

void UseDefChain::insertTrainingDef(const TrainingOperationIndex &idx)
{
  // defs must be valid
  assert(idx.valid());
  _defs.insert(idx);
}

void UseDefChain::removeTrainingDef(const TrainingOperationIndex &idx) { _defs.erase(idx); }

void UseDefChain::clearTrainingUseDefs()
{
  _uses.clear();
  _defs.clear();
}

bool UseDefChain::operator==(const UseDefChain &other) const
{
  return &_operand == &other._operand && _uses == other._uses && _defs == other._defs;
}

} // namespace onert::ir::train
