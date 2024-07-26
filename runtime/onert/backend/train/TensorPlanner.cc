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

void TensorPlanner::planNonConstTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(BackendContext) << "Start planning non-constant tensors" << std::endl;

  const auto &training_usedefs = _tgraph.trainingUseDefs();

  // NOTE The uses_map and defs_map must have the size of only registered tensors
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;

  // Prepare scanning
  // This assumes TrainingOperationIndex in forwarding are always used
  for (const auto &[operand_index, operand_usedefs] : training_usedefs)
  {
    const auto &operand = operand_usedefs.operand();

    if (_external_operands.contains(operand_index.index()))
      continue;

    if (!operand_index.is_forward() || operand.isConstant())
      continue;

    uses_map[operand_index] = operand_usedefs.getTrainingUses().size();
    defs_map[operand_index] = operand_usedefs.getTrainingDefs().size();
  }

  // Start scanning to do notify{First|Last}Use for each tensor
  // TODO Remove this or find the reason why it is needed
  // Q. Why is notifyFirstUse() called if operand's def count is 0?
  //    It's neither an external operand or a constant operand
  //    What does it mean when def count is 0?
  // A. Not yet found the reason to need it yet.
  for (const auto &[operand_index, def_count] : defs_map)
  {
    if (def_count == 0)
      tensor_builder->notifyFirstUse(operand_index.index());
  }

  // This is a workaround to keep the operands over the execution
  // (the operands look like they are unused)
  std::vector<ir::train::TrainingOperandIndex> operands_last_until_end;
  for (const auto &[operand_index, use_count] : uses_map)
  {
    if (use_count == 0)
      operands_last_until_end.push_back(operand_index);
  }

  // Plan used or defined tensors in forwarding nodes
  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, throw an exception (not supported yet)
  // 3. Scan USE of inputs/outputs. Decrease the USE and deallocate if the USE is 0
  const auto order = _tgraph.topolSortOperations();
  for (const auto &op_index : order)
  {
    const auto &op = _tgraph.operations().at(op_index);
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
    auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    // Define outputs
    for (const auto &output : op_outputs)
    {
      if (_external_operands.contains(output))
        continue;
      if (!tensor_builder->isRegistered(output))
        continue;

      const auto output_index = ir::train::TrainingOperandIndex{output, true};
      assert(defs_map.find(output_index) != defs_map.end());
      assert(defs_map.at(output_index) == 1);
      defs_map[output_index] = 0;
      tensor_builder->notifyFirstUse(output_index.index());
    }

    // Scan variable tensors
    // This tensor has features like constant. But OperandInfo and LowerInfo treat them as
    // non-constant because of less memory usage by memory planning in here
    // However, train backend does not support variable tensors yet
    for (const auto &input : op_inputs)
    {
      if (_external_operands.contains(input))
        continue;
      if (!tensor_builder->isRegistered(input))
        continue;

      const auto input_index = ir::train::TrainingOperandIndex{input, true};
      const auto &operand = training_usedefs.at(input_index).operand();
      if (operand.isConstant())
        continue;

      assert(training_usedefs.find(input_index) != training_usedefs.end());
      if (operand.info().isVariable())
        throw std::runtime_error("The train backend does not support variable tensors");
    }

    for (const auto &input : op_inputs)
    {
      if (_external_operands.contains(input))
        continue;
      if (!tensor_builder->isRegistered(input))
        continue;

      const auto input_index = ir::train::TrainingOperandIndex{input, true};
      const auto &operand = training_usedefs.at(input_index).operand();
      if (operand.isConstant())
        continue;

      assert(uses_map.find(input_index) != uses_map.end());
      assert(uses_map[input_index] > 0);
      uses_map[input_index]--;
      if (uses_map[input_index] == 0)
      {
        // plan for deallocation of static tensor node
        tensor_builder->notifyLastUse(input_index.index());
      }
    }
  }

  // Plan used tensors in backwarding nodes
  const auto border = _tgraph.essentialBackwardOrder();
  for (const auto &op_index : border)
  {
    const auto &op = _tgraph.operations().at(op_index);
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
    auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    for (const auto &index : op_inputs + op_outputs)
    {
      if (_external_operands.contains(index))
        continue;
      if (!tensor_builder->isRegistered(index))
        continue;

      const auto operand_index = ir::train::TrainingOperandIndex{index, true};
      assert(training_usedefs.find(operand_index) != training_usedefs.end());
      const auto &operand_usedefs = training_usedefs.at(operand_index);
      const auto &operand = operand_usedefs.operand();
      if (operand.isConstant())
        continue;

      const auto &training_op_index = ir::train::TrainingOperationIndex{op_index, false};
      assert(operand_usedefs.getTrainingDefs().find(training_op_index) ==
             operand_usedefs.getTrainingDefs().end());

      const auto &uses = operand_usedefs.getTrainingUses();
      if (uses.find(training_op_index) != uses.end())
      {
        assert(uses_map.find(operand_index) != uses_map.end());
        assert(uses_map[operand_index] > 0);
        uses_map[operand_index]--;
        if (uses_map[operand_index] == 0)
        {
          // plan for deallocation of static tensor node
          tensor_builder->notifyLastUse(operand_index.index());
        }
      }
    }
  }

  for (const auto &operand_index : operands_last_until_end)
  {
    tensor_builder->notifyLastUse(operand_index.index());
  }

  assert(std::all_of(
    uses_map.begin(), uses_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  assert(std::all_of(
    defs_map.begin(), defs_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  VERBOSE(BackendContext) << "Finish planning non-constant tensors" << std::endl;
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
