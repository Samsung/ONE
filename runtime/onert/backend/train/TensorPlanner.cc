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

namespace onert::backend::train
{

TensorPlanner::TensorPlanner(const ir::train::TrainableGraph &tgraph,
                             const util::Set<ir::OperandIndex> &external_operands)
  : _tgraph{tgraph}, _external_operands{external_operands}
{
  // DO NOTHING
}

void TensorPlanner::planNonConstTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning non-constant tensors" << std::endl;

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
    auto op_inputs = op.getUsedInputSet();
    auto op_outputs = op.getUsedOutputSet();

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
    auto op_inputs = op.getUsedInputSet();
    auto op_outputs = op.getUsedOutputSet();

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

  VERBOSE(TensorPlanner) << "Finish planning non-constant tensors" << std::endl;
}

void TensorPlanner::planTrainableTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning constant tensors" << std::endl;

  const auto &training_usedefs = _tgraph.trainingUseDefs();

  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;
  std::vector<ir::train::TrainingOperandIndex> constants;

  // Prepare scanning
  for (const auto &[operand_index, operand_usedefs] : training_usedefs)
  {
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

  VERBOSE(TensorPlanner) << "Finish planning constant tensors" << std::endl;
}

void TensorPlanner::planBackPropTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning back-propagated tensors" << std::endl;

  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;

  // Prepare scanning
  const auto &training_usedefs = _tgraph.trainingUseDefs();
  for (const auto &[operand_index, operand_usedefs] : training_usedefs)
  {
    const auto &operand = operand_usedefs.operand();

    if (_external_operands.contains(operand_index.index()))
      continue;

    if (!tensor_builder->isRegisteredBackward(operand_index.index()))
      continue;

    if (operand_index.is_forward() || operand.isConstant())
      continue;

    uses_map[operand_index] = operand_usedefs.getTrainingUses().size();
    defs_map[operand_index] = operand_usedefs.getTrainingDefs().size();
  }

  // Start scanning to do notify{First|Last}Use for each tensor

  // This is a workaround to keep the operands over the execution
  // (the operands look like they are unused)
  std::vector<ir::train::TrainingOperandIndex> operands_last_until_end;
  for (const auto &[ind, use_count] : uses_map)
  {
    if (use_count == 0)
      operands_last_until_end.push_back(ind);
  }

  // At each operation,
  // 1. Scan DEF of outgoing tnesors. If the first DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, throw an exception (not supported yet)
  // 3. Scan USE of incoming tensors. Decrease the USE and deallocate if the USE is 0
  std::set<ir::OperandIndex> unallocated;
  _tgraph.operands().iterate(
    [&](const ir::OperandIndex &index, const ir::Operand &) { unallocated.insert(index); });

  const auto border = _tgraph.essentialBackwardOrder();
  for (const auto &op_ind : border)
  {
    const auto &op = _tgraph.operations().at(op_ind);
    auto op_inputs = op.getUsedInputSet();
    auto op_outputs = op.getUsedOutputSet();

    // Allocate back-propagated tensors in first def
    for (const auto &outgoing : op_inputs)
    {
      const auto operand_index = ir::train::TrainingOperandIndex{outgoing, false};
      const auto &operand = _tgraph.operands().at(outgoing);
      if (_external_operands.contains(outgoing))
        continue;
      if (!tensor_builder->isRegisteredBackward(outgoing))
        continue;
      if (operand.isConstant())
        continue;

      if (defs_map.find(operand_index) != defs_map.end())
      {
        if (unallocated.find(outgoing) != unallocated.end())
        {
          // First Def
          unallocated.erase(outgoing);
          defs_map[operand_index]--;
          tensor_builder->notifyBackwardFirstUse(outgoing);
        }
        else
        {
          assert(defs_map[operand_index] > 0);
          defs_map[operand_index]--;
        }
      }
    }

    // Scan variable tensors
    // This tensor has features like constant. But OperandInfo and LowerInfo treat them as
    // non-constant because of less memory usage by memory planning in here
    // However, train backend does not support variable tensors yet
    for (const auto &outgoing : op_inputs)
    {
      if (_external_operands.contains(outgoing))
        continue;
      if (!tensor_builder->isRegisteredBackward(outgoing))
        continue;
      const auto &operand = _tgraph.operands().at(outgoing);
      if (operand.info().isVariable())
        throw std::runtime_error("The train backend does not support variable tensors");
    }

    for (const auto &incoming : op_outputs)
    {
      const auto incoming_index = ir::train::TrainingOperandIndex{incoming, false};

      if (_external_operands.contains(incoming))
        continue;
      if (!tensor_builder->isRegisteredBackward(incoming))
        continue;

      // NOTE There is no case where an op's incoming tensors don't have the corresponding op def
      assert(defs_map.find(incoming_index) != defs_map.end());

      if (uses_map.find(incoming_index) != uses_map.end())
      {
        assert(uses_map[incoming_index] > 0);
        uses_map[incoming_index]--;
        if (uses_map[incoming_index] == 0)
        {
          // plan for deallocation of static tensornode
          tensor_builder->notifyBackwardLastUse(incoming);
        }
      }
    }
  }

  for (const auto &index : operands_last_until_end)
  {
    tensor_builder->notifyBackwardLastUse(index.index());
  }

  assert(std::all_of(
    uses_map.begin(), uses_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  assert(std::all_of(
    defs_map.begin(), defs_map.end(),
    [](std::pair<const ir::train::TrainingOperandIndex, uint32_t> it) { return it.second == 0; }));

  VERBOSE(TensorPlanner) << "Finish planning back-propagated tensors" << std::endl;
}

void TensorPlanner::planGradientTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning gradient tensors" << std::endl;

  // TODO Use DisposableTensor instead of GradientTensor to plan them together if possible
  //      Backward layers and the corresponding GradientApplier exist in the same back-propagated
  //      operation sequence. So we can use DisposableTensors to plan GradientTensors.
  for (const auto &op_index : _tgraph.essentialBackwardOrder())
  {
    std::vector<ir::train::TrainingOperandIndex> cur_seq;
    const auto &op = _tgraph.operations().at(op_index);
    const auto backwarding_op_index = ir::train::TrainingOperationIndex{op_index, false};
    auto op_inputs = op.getUsedInputSet();

    // Only inputs can be candidates for def of backwarding tensors
    for (const auto &input : op_inputs)
    {
      if (_external_operands.contains(input))
        continue;
      if (!tensor_builder->isRegisteredBackward(input))
        continue;

      const auto gradient_index = ir::train::TrainingOperandIndex{input, false};
      const auto &training_usedefs = _tgraph.trainingUseDefs();
      const auto &usedefs = training_usedefs.at(gradient_index);
      const auto &operand = usedefs.operand();
      const auto &defs = usedefs.getTrainingDefs();
      if (operand.isConstant() && defs.find(backwarding_op_index) != defs.end())
      {
        assert(defs.size() == 1);
        tensor_builder->notifyBackwardFirstUse(input);
        cur_seq.emplace_back(gradient_index);
      }
    }

    for (const auto &operand_index : cur_seq)
    {
      tensor_builder->notifyBackwardLastUse(operand_index.index());
    }
  }

  VERBOSE(TensorPlanner) << "Finish planning gradient tensors" << std::endl;
}

void TensorPlanner::planDisposableBackPropTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning disposable back-prop tensors" << std::endl;

  for (const auto &op_index : _tgraph.essentialBackwardOrder())
  {
    // NOTE Even if there are duplicate indices, the duplicate back-propagated tensors may need
    //      to be updated respectively. So we use a sequence instead of a set.
    const auto &inputs = _tgraph.operation(op_index).getInputs();
    if (!(inputs == (inputs | ir::Remove::DUPLICATED)))
      throw std::runtime_error("TensorPlanner: DispoableBackProp tensor does not support duplicate "
                               "inputs of an operation");

    std::vector<DisposableTensorIndex> cur_seq;
    const auto back_prop_indices = getOutgoingBackPropSeq(op_index, tensor_builder);
    for (const auto &back_prop_index : back_prop_indices)
    {
      DisposableTensorIndex cur_index{op_index, back_prop_index};
      if (tensor_builder->isRegisteredDisposableBackwardTensor(cur_index))
      {
        tensor_builder->notifyDisposableBackPropFirstUse(cur_index);
        cur_seq.emplace_back(cur_index);
      }
    }

    for (const auto &cur_index : cur_seq)
    {
      tensor_builder->notifyDisposableBackPropLastUse(cur_index);
    }
  }

  VERBOSE(TensorPlanner) << "Finish planning disposable back-prop tensors" << std::endl;
}

ir::OperandIndexSequence TensorPlanner::getOutgoingBackPropSeq(const ir::OperationIndex &op_index,
                                                               const TensorBuilder *tensor_builder)
{
  ir::OperandIndexSequence ret;

  const auto &op = _tgraph.operation(op_index);
  for (const auto &input : op.getUsedInputSet())
  {
    if (_external_operands.contains(input))
      continue;
    if (!tensor_builder->isRegisteredBackward(input))
      continue;

    const auto input_index = ir::train::TrainingOperandIndex{input, false};
    const auto training_op_index = ir::train::TrainingOperationIndex{op_index, false};
    const auto &training_usedefs = _tgraph.trainingUseDefs();
    const auto &usedefs = training_usedefs.at(input_index);
    if (usedefs.operand().isConstant())
      continue;

    if (usedefs.getTrainingDefs().find(training_op_index) == usedefs.getTrainingDefs().end())
      continue;

    ret.append(input);
  }

  return ret;
}

void TensorPlanner::planLayerScopeTensors(TensorBuilder *tensor_builder)
{
  VERBOSE(TensorPlanner) << "Start planning layer scope tensors" << std::endl;

  // forwading order
  const auto f_order = _tgraph.topolSortOperations();
  for (const auto &op_index : f_order)
  {
    if (not tensor_builder->isRegisteredLayerScopeTensor(op_index))
      continue;

    const auto &indices = tensor_builder->getRegisteredLayerScopeTensorIndices(op_index);
    for (const auto &idx : indices)
    {
      const auto lt = tensor_builder->getLayerScopeTensorLifeTime(idx);
      if (lt == LayerScopeTensorLifeTime::FORWARD_TO_BACKWARD)
        tensor_builder->notifyLayerScopeFirstUse(idx);
    }
  }

  // backwarding order
  const auto b_order = _tgraph.essentialBackwardOrder();
  for (const auto &op_index : b_order)
  {
    if (not tensor_builder->isRegisteredLayerScopeTensor(op_index))
      continue;

    const auto &indices = tensor_builder->getRegisteredLayerScopeTensorIndices(op_index);

    for (const auto &idx : indices)
    {
      const auto lt = tensor_builder->getLayerScopeTensorLifeTime(idx);
      if (lt == LayerScopeTensorLifeTime::BACKWARD)
        tensor_builder->notifyLayerScopeFirstUse(idx);
    }
    for (const auto &idx : indices)
    {
      const auto lt = tensor_builder->getLayerScopeTensorLifeTime(idx);
      if (lt == LayerScopeTensorLifeTime::FORWARD_TO_BACKWARD ||
          lt == LayerScopeTensorLifeTime::BACKWARD)
        tensor_builder->notifyLayerScopeLastUse(idx);
    }
  }

  VERBOSE(TensorPlanner) << "Finish planning layerscope tensors" << std::endl;
}

} // namespace onert::backend::train
