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

#ifndef __ONERT_IR_TRAIN_DEFUSECHAIN_H__
#define __ONERT_IR_TRAIN_DEFUSECHAIN_H__

#include "ir/Operand.h"
#include "ir/train/Index.h"

#include <set>

namespace onert
{
namespace ir
{
namespace train
{

class DefUseChain
{
public:
  explicit DefUseChain(const Operand &operand) : _operand{operand}
  {
    // DO NOTHING
  }
  explicit DefUseChain(const DefUseChain &other)
    : _operand{other._operand}, _uses{other._uses}, _def{other._def}
  {
    // DO NOTHING
  }
  DefUseChain(DefUseChain &&other)
    : _operand{other._operand}, _uses{std::move(other._uses)}, _def{std::move(other._def)}
  {
    // DO NOTHING
  }
  ~DefUseChain() = default;

public:
  DefUseChain &operator=(const DefUseChain &other) = delete;
  DefUseChain &operator=(DefUseChain &&other) = delete;

public:
  bool operator==(const DefUseChain &other) const;

public:
  const std::set<TrainingOperationIndex> &getTrainingUses() const { return _uses; }
  TrainingOperationIndex getTrainingDef() const { return _def; }
  void insertTrainingUse(const TrainingOperationIndex &idx);
  void removeTrainingUse(const TrainingOperationIndex &idx);
  void setTrainingDef(const TrainingOperationIndex &idx);
  void unsetTrainingDef();
  void clearTrainingDefUse();

public:
  const Operand &operand() const { return _operand; };

private:
  const Operand &_operand;

private:
  std::set<TrainingOperationIndex> _uses;
  TrainingOperationIndex _def;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_DEFUSECHAIN_H__
