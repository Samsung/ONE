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

#include "TensorPlanner.h"

#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace train
{

TensorPlanner::TensorPlanner(const ir::train::TrainableGraph &tgraph,
                             const util::Set<ir::OperandIndex> &external_operands)
  : _tgraph{tgraph}, _external_operands{external_operands}
{
  // DO NOTHING
  // TODO Remove the following lines
  UNUSED_RELEASE(_tgraph);
  UNUSED_RELEASE(_external_operands);
}

void TensorPlanner::planNonConstTensors(TensorBuilder *)
{
  // TODO Plan non-const tensors
}

void TensorPlanner::planTrainableTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(BackendContext) << "Start planning constant tensors" << std::endl;

  const auto &training_usedefs = _tgraph.trainingUseDefs();

  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;
  std::vector<ir::train::TrainingOperandIndex> constants;

  // Prepare scanning
  for (const auto &pair : training_usedefs)
  {
    const auto &operand_index = pair.first;
    const auto &operand_usedefs = pair.second;
    const auto &operand = operand_usedefs.operand();

    if (!operand_index.valid())
      continue;

    if (operand.isConstant() && operand_index.is_forward())
    {
      uses_map[operand_index] = 0;
      const auto &defs = operand_usedefs.getTrainingDefs();
      defs_map[operand_index] = defs.size(); // It means def_map's values are 0
      constants.emplace_back(operand_index);
    }
  }

  // Start scanning to do notify{First|Last}Use for each tensor
  // If a tensor is a constant, increase the use of the tensor and allocate it first.
  // Increasing use count here makes the tensor never be deallocated, i.e it they will be
  // deallocated last.
  for (const auto &index : constants)
  {
    assert(index.is_forward());
    if (tensor_builder->isRegistered(index.index()))
    {
      uses_map[index]++;
      tensor_builder->notifyFirstUse(index.index());
    }
  }

  // Dispose and validate
  for (const auto &index : constants)
  {
    assert(index.is_forward());
    if (tensor_builder->isRegistered(index.index()))
    {
      uses_map[index]--;
      tensor_builder->notifyLastUse(index.index());
    }
  }

  assert(std::all_of(
    uses_map.begin(), uses_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  assert(std::all_of(
    defs_map.begin(), defs_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  VERBOSE(BackendContext) << "Finish planning constant tensors" << std::endl;
}

void TensorPlanner::planBackPropTensors(TensorBuilder *)
{
  // TODO Plan back-propagated tensors
}

void TensorPlanner::planGradientTensors(TensorBuilder *)
{
  // TODO Plan gradient tensors
}

void TensorPlanner::planDisposableBackPropTensors(TensorBuilder *)
{
  // TODO Plan diposable backprop tensors
}

} // namespace train
} // namespace backend
} // namespace onert
