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

#include "ir/train/DefUseChain.h"

#include <algorithm>
#include <memory>

namespace onert
{
namespace ir
{
namespace train
{

void DefUseChain::insertTrainingUse(const TrainingOperationIndex &idx) { _uses.insert(idx); }

void DefUseChain::removeTrainingUse(const TrainingOperationIndex &idx) { _uses.erase(idx); }

void DefUseChain::setTrainingDef(const TrainingOperationIndex &idx) { _def = idx; }

void DefUseChain::unsetTrainingDef() { _def = TrainingOperationIndex{}; }

void DefUseChain::clearTrainingDefUse()
{
  unsetTrainingDef();
  _uses.clear();
}

bool DefUseChain::operator==(const DefUseChain &other) const
{
  return _uses == other._uses && _def == other._def;
}

} // namespace train
} // namespace ir
} // namespace onert
