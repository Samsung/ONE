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

#ifndef __ONERT_IR_TRAIN_USEDEFCHAIN_H__
#define __ONERT_IR_TRAIN_USEDEFCHAIN_H__

#include "ir/Operand.h"
#include "ir/train/Index.h"

#include <set>

namespace onert::ir::train
{

class UseDefChain
{
public:
  explicit UseDefChain(const Operand &operand) : _operand{operand}
  {
    // DO NOTHING
  }
  explicit UseDefChain(const UseDefChain &other)
    : _operand{other._operand}, _uses{other._uses}, _defs{other._defs}
  {
    // DO NOTHING
  }
  UseDefChain(UseDefChain &&other)
    : _operand{other._operand}, _uses{std::move(other._uses)}, _defs{std::move(other._defs)}
  {
    // DO NOTHING
  }
  ~UseDefChain() = default;

public:
  UseDefChain &operator=(const UseDefChain &other) = delete;
  UseDefChain &operator=(UseDefChain &&other) = delete;

public:
  bool operator==(const UseDefChain &other) const;

public:
  const std::set<TrainingOperationIndex> &getTrainingUses() const { return _uses; }
  const std::set<TrainingOperationIndex> &getTrainingDefs() const { return _defs; }
  void insertTrainingUse(const TrainingOperationIndex &idx);
  void removeTrainingUse(const TrainingOperationIndex &idx);
  void insertTrainingDef(const TrainingOperationIndex &idx);
  void removeTrainingDef(const TrainingOperationIndex &idx);
  void clearTrainingUseDefs();

public:
  const Operand &operand() const { return _operand; };

private:
  const Operand &_operand;

private:
  std::set<TrainingOperationIndex> _uses;
  // NOTE Allowing multiple defs is a workaround to support training of branching models.
  //      Back-prop tensors corresponding to forwarding tensors that are used in multiple nodes
  //      have multiple defs. Those back-prop tensors would be accumulated during backwarding,
  //      but current TrainableGraph cannot represent accumulated back-prop tensors.
  std::set<TrainingOperationIndex> _defs;
};

} // namespace onert::ir::train

#endif // __ONERT_IR_TRAIN_USEDEFCHAIN_H__
