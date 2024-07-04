/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BackendContext.h"

#include "TensorBuilder.h"
#include "KernelGenerator.h"
#include "ops/BackPropInitializer.h"
#include "ops/GradientApplier.h"

#include <backend/basic/train/TrainableBackendContextHelpers.h>
#include <misc/polymorphic_downcast.h>

#include <cassert>

namespace onert
{
namespace backend
{
namespace train
{

namespace
{
ir::OperandInfo createBackwardTensorInfo(const ir::Operand &operand)
{
  // TODO Use different shape of back-propagated tensor if it exists
  return ir::OperandInfo{operand.shape(), operand.typeInfo(), operand.info().memAllocType(),
                         operand.isConstant()};
}

// NOTE Even if there are duplicate indices, the duplicate back-propagated tensors may need
//      to be updated respectively. So we use a sequence instead of a set.
ir::OperandIndexSequence getBackPropSeq(const ir::train::TrainableGraph &tgraph,
                                        const ir::OperationIndex &op_index)
{
  ir::OperandIndexSequence ret;

  const auto &op = tgraph.operations().at(op_index);
  for (const auto &input : (op.getInputs() | ir::Remove::UNDEFINED))
  {
    const auto &operand = tgraph.operands().at(input);
    // TODO Remove other inputs that are not back-propagated
    if (!operand.isConstant() && !tgraph.getInputs().contains(input))
      ret.append(input);
  }

  return ret;
}

void AddBackPropInitializers(const ir::train::TrainableGraph &tgraph, TensorRegistry &tensor_reg,
                             FunctionMap &fn_map)
{
  util::Set<ir::OperandIndex> unvisited;
  tgraph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &operand) {
    if (!tgraph.getInputs().contains(index) && !operand.isConstant())
      unvisited.add(index);
  });

  for (const auto &op_index : tgraph.getEssentialBackwardOrder())
  {
    assert(fn_map.find(op_index) != fn_map.end());

    auto &tn_seq = fn_map.at(op_index);

    // The function added latest is executed first in a sequence during backwarding.
    std::vector<BackPropTensor *> back_props;
    const auto &op = tgraph.operation(op_index);
    for (const auto &back_prop_index :
         op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
    {
      assert(op.isRequiredForBackward());
      if (unvisited.contains(back_prop_index))
      {
        auto back_prop_tensor = tensor_reg.getBackPropTensor(back_prop_index);
        assert(back_prop_tensor != nullptr);
        back_props.emplace_back(back_prop_tensor);
        unvisited.remove(back_prop_index);
      }
    }

    if (back_props.size() != 0)
    {
      auto initializer = std::make_unique<ops::BackPropInitializer>(back_props);
      tn_seq->append(std::move(initializer));
    }
  }
}

std::set<ir::train::TrainingOperandIndex>
getBackwardTensorList(const ir::train::TrainableGraph &tgraph, const BackendContext &ctx)
{
  std::set<ir::train::TrainingOperandIndex> ret;

  // TODO Reuse registered tensors when they are planned for memory optimization.
  auto border = tgraph.getEssentialBackwardOrder();
  for (const auto op_index : border)
  {
    const auto &trainable_op = tgraph.operation(op_index);
    assert(trainable_op.isRequiredForBackward());
    // This assumes that back-propagated tensors of loss outputs are not used
    for (const auto &ind :
         trainable_op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      if (ctx.external_operands().contains(ind))
        continue;

      const auto &operand_index = ir::train::TrainingOperandIndex{ind, false};

      const auto &training_usedefs = tgraph.trainingUseDefs();
      const auto &usedefs = training_usedefs.at(ir::train::TrainingOperandIndex{ind, false});
      const bool not_used = usedefs.getTrainingDefs().empty() && usedefs.getTrainingUses().empty();
      if (not_used)
        continue;

      ret.insert(operand_index);
    }
  }

  return ret;
}

/**
 * @brief  Plan constant tensors to optimize memory
 */
void planConstTensors(BackendContext &ctx)
{
  VERBOSE(BackendContext) << "Start planning constant tensors" << std::endl;

  const ir::train::TrainableGraph &tgraph = *ctx.data()->tgraph;

  const auto &training_usedefs = tgraph.trainingUseDefs();

  auto tensor_builder = ctx.tensor_builder();

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
      // const auto &info = operand.info();
      // tensor_builder->registerTensorInfo(operand_index.index(), info, ir::Layout::NHWC);

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

void planNonConstTensors(BackendContext &ctx)
{
  VERBOSE(BackendContext) << "Start planning non-constant tensors" << std::endl;

  const ir::train::TrainableGraph &tgraph = *ctx.data()->tgraph;

  const auto &training_usedefs = tgraph.trainingUseDefs();

  auto tensor_builder = ctx.tensor_builder();

  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;
  // ir::OperandIndexSequence nonconstants;

  // Prepare scanning
  // uses_map and defs_map must have size of TrainindOperandIndex list of only registered tensors
  // TrainingOperationIndex in forwarding are always used
  // const auto used_backward_indices = getBackwardTensorList(tgraph, ctx);
  for (const auto &pair : training_usedefs)
  {
    const auto &operand_index = pair.first;
    const auto &operand_usedefs = pair.second;
    const auto &operand = operand_usedefs.operand();

    if (ctx.external_operands().contains(operand_index.index()))
      continue;

    if (!operand_index.is_forward() || operand.isConstant())
      continue;

    uses_map[operand_index] = operand_usedefs.getTrainingUses().size();
    defs_map[operand_index] = operand_usedefs.getTrainingDefs().size();

    // assert(operand_index.is_forward());
    // if (!tensor_builder->isRegistered(operand_index.index()))
    // {
    //   // These tensors do not exist in any  (No use and def)
    //   const auto &info = operand.info();
    //   // NOTE Currently we only support NHWC tensors for cpu-common tensors.
    //   //      There is no way to get the layout info from the backend context for now.
    //   //      When we support NCHW tensors as well, we also need to change tensor info to be
    //   //      permuted shape.
    //   assert(ctx.operand_layouts().at(operand_index.index()) == ir::Layout::NHWC);
    //   tensor_builder->registerTensorInfo(operand_index.index(), info, ir::Layout::NHWC);
    // }
  }

  // TODO Remove this or find the reason why it is needed
  // Start scanning to do notify{First|Last}Use for each tensor
  // Q. Why is notifyFirstUse() called if the operand's def count is 0? Is it not a constant
  // operand?
  //    What does it mean when def count is 0?
  // A. No answer yet.
  for (const auto &pair : defs_map)
  {
    const auto &operand_index = pair.first;
    const auto def_count = pair.second;
    if (def_count == 0)
      tensor_builder->notifyFirstUse(operand_index.index());
  }

  // This is a workaround to keep the operands over the execution
  // (the operands look like they are unused)
  std::vector<ir::train::TrainingOperandIndex> operands_last_until_end;
  for (const auto &pair : uses_map)
  {
    const auto &operand_index = pair.first;
    const auto use_count = pair.second;
    if (use_count == 0)
      operands_last_until_end.push_back(operand_index);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, throw an exception (not supported yet)
  // 3. Scan USE of inputs/outputs. Decrease the USE and deallocate if the USE is 0
  const auto &order = ctx.data()->op_order;
  assert(order == tgraph.topolSortOperations());
  for (const auto &op_index : order)
  {
    const auto &op = tgraph.operations().at(op_index);
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
    auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    // Define outputs
    for (const auto &output : op_outputs)
    {
      if (ctx.external_operands().contains(output))
        continue;
      if (!tensor_builder->isRegistered(output))
        continue;

      const auto output_index = ir::train::TrainingOperandIndex{output, true};
      assert(defs_map.find(output_index) != defs_map.end());
      defs_map[output_index] = 0;
      tensor_builder->notifyFirstUse(output_index.index());
    }

    // Scan variable tensors
    // This tensor has features like constant. But OperandInfo and LowerInfo treat them as
    // non-constant because of less memory usage by memory planning in here
    // However, train backend does not support variable tensors yet
    for (const auto &input : op_inputs)
    {
      if (ctx.external_operands().contains(input))
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
      if (ctx.external_operands().contains(input))
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

  const auto border = tgraph.getEssentialBackwardOrder();
  for (const auto &op_index : border)
  {
    const auto &op = tgraph.operations().at(op_index);
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
    auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    for (const auto &index : op_inputs + op_outputs)
    {
      if (ctx.external_operands().contains(index))
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

// Plan GradientTensors to keep from the corresponding backward layer to the corresponding
void planGradientTensors(BackendContext &ctx)
{
  VERBOSE(BackendContext) << "Start planning gradient tensors" << std::endl;
  // TODO Apply DisposableTensor instead of GradientTensor if possible
  // The corresponding backward layer and the corresponding GradientApplier exist in the same
  // back-propagated operation sequence. So we can use DisposableTensors to plan GradientTensors.
  const ir::train::TrainableGraph &tgraph = *ctx.data()->tgraph;
  auto tensor_builder = ctx.tensor_builder();

  std::vector<ir::train::TrainingOperandIndex> prev_seq;
  for (const auto &op_index : tgraph.getEssentialBackwardOrder())
  {
    for (const auto &operand_index : prev_seq)
    {
      tensor_builder->notifyBackwardLastUse(operand_index.index());
    }

    std::vector<ir::train::TrainingOperandIndex> cur_seq;
    const auto &op = tgraph.operations().at(op_index);
    const auto backwarding_op_index = ir::train::TrainingOperationIndex{op_index, false};
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    // Only inputs can be candidates for def of backwarding tensors
    for (const auto &input : op_inputs)
    {
      if (ctx.external_operands().contains(input))
        continue;
      if (!tensor_builder->isRegisteredBackward(input))
        continue;

      const auto input_index = ir::train::TrainingOperandIndex{input, false};
      const auto &training_usedefs = tgraph.trainingUseDefs();
      const auto &usedefs = training_usedefs.at(input_index);
      const auto &operand = usedefs.operand();
      const auto &defs = usedefs.getTrainingDefs();
      if (operand.isConstant() && defs.find(backwarding_op_index) != defs.end())
      {
        assert(defs.size() == 1);
        tensor_builder->notifyBackwardFirstUse(input);
        cur_seq.emplace_back(input_index);
      }
    }

    prev_seq = cur_seq;
  }
  VERBOSE(BackendContext) << "Finish planning gradient tensors" << std::endl;
}

// From the begining of backward to the end of backward,
// BackProp tensors are available to have multiple defs
void planBackPropTensors(BackendContext &ctx)
{
  VERBOSE(BackendContext) << "Start planning back-propagated tensors" << std::endl;
  const ir::train::TrainableGraph &tgraph = *ctx.data()->tgraph;

  auto tensor_builder = ctx.tensor_builder();

  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> uses_map;
  std::unordered_map<ir::train::TrainingOperandIndex, uint32_t> defs_map;
  // ir::OperandIndexSequence constants;

  // Prepare scanning
  const auto &training_usedefs = tgraph.trainingUseDefs();
  for (const auto &pair : training_usedefs)
  {
    const auto &operand_index = pair.first;
    const auto &operand_usedefs = pair.second;
    const auto &operand = operand_usedefs.operand();

    if (ctx.external_operands().contains(operand_index.index()))
      continue;

    if (!tensor_builder->isRegisteredBackward(operand_index.index()))
      continue;

    if (operand_index.is_forward() || operand.isConstant())
      continue;

    // TODO Check if we need to handle unused tensors

    uses_map[operand_index] = operand_usedefs.getTrainingUses().size();
    defs_map[operand_index] = operand_usedefs.getTrainingDefs().size();

    //   if ((operand_index.is_forward() && !tensor_builder->isRegistered(operand_index.index())) &&
    //       (!operand_index.is_forward() &&
    //       !tensor_builder->isRegisteredBackward(operand_index.index())))
    //   {
    //     // These tensors do not exist in any  (No use and def)
    //     const auto &info = operand.info();
    //     // NOTE Currently we only support NHWC tensors for cpu-common tensors.
    //     //      There is no way to get the layout info from the backend context for now.
    //     //      When we support NCHW tensors as well, we also need to change tensor info to be
    //     //      permuted shape.
    //     assert(ctx.operand_layouts().at(operand_index.index()) == ir::Layout::NHWC);
    //     tensor_builder->registerTensorInfo(operand_index.index(), info, ir::Layout::NHWC);
    //   }
  }

  // Start scanning to do notify{First|Last}Use for each tensor

  // If a tensor is a constant, increase the use of the tensor and allocate it first.
  // Increasing use count here makes the tensor never be deallocated, i.e it they will be
  // deallocated last.
  // for (const auto &index : constants)
  // {
  //   uses_map[ir::train::TrainingOperandIndex{index, true}]++;
  //   tensor_builder->notifyFirstUse(index);
  // }

  // for (const auto &pair : defs_map)
  // {
  //   const auto &index = pair.first;
  //   const auto def_count = pair.second;
  //   if (def_count == 0)
  //     tensor_builder->notifyFirstUse(index);
  // }

  // This is a workaround to keep the operands over the execution
  // (the operands look like they are unused)
  std::vector<ir::train::TrainingOperandIndex> operands_last_until_end;
  for (const auto &pair : uses_map)
  {
    const auto &ind = pair.first;
    const auto use_count = pair.second;
    if (use_count == 0)
      operands_last_until_end.push_back(ind);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the first DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, throw an exception (not supported yet)
  // 3. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  std::set<ir::OperandIndex> unallocated;
  tgraph.operands().iterate(
    [&](const ir::OperandIndex &index, const ir::Operand &) { unallocated.insert(index); });

  const auto border = tgraph.getEssentialBackwardOrder();
  for (const auto &op_ind : border)
  {
    const auto &op = tgraph.operations().at(op_ind);
    auto op_inputs = op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
    auto op_outputs = op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;

    // Allocate back-propagated tensors in first def
    for (const auto &outgoing : op_inputs)
    {
      const auto operand_index = ir::train::TrainingOperandIndex{outgoing, false};
      const auto &operand = tgraph.operands().at(outgoing);
      if (ctx.external_operands().contains(outgoing))
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
      if (ctx.external_operands().contains(outgoing))
        continue;
      if (!tensor_builder->isRegisteredBackward(outgoing))
        continue;
      const auto &operand = tgraph.operands().at(outgoing);
      if (operand.info().isVariable())
        throw std::runtime_error("The train backend does not support variable tensors");
    }

    for (const auto &incoming : op_outputs)
    {
      const auto incoming_index = ir::train::TrainingOperandIndex{incoming, false};

      if (ctx.external_operands().contains(incoming))
        continue;
      if (!tensor_builder->isRegisteredBackward(incoming))
        continue;

      // NOTE There is no case where an op's incoming tensors have the coresponding op def yet
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

  VERBOSE(BackendContext) << "Finish planning back-propagated tensors" << std::endl;
}

} // namespace

backend::ITensorRegistry *BackendContext::genTensors()
{
  const auto &tgraph = *trainable_graph();

  tgraph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &obj) {
    if (external_operands().contains(index))
      return;
    if (!index.valid())
      return;

    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);
    _tensor_builder->registerTensorInfo(index, obj.info(), ir::Layout::NHWC);
  });

  planConstTensors(*this);
  planNonConstTensors(*this);

  _tensor_builder->allocate();

  return _tensor_registry.get();
}

backend::train::ITensorRegistry *BackendContext::genTrainingTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;

  const auto operand_indices = getBackwardTensorList(tgraph, *this);
  for (const auto &operand_index : operand_indices)
  {
    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);
    assert(!operand_index.is_forward());
    const auto &operand = tgraph.operands().at(operand_index.index());
    tensor_builder->registerBackwardTensorInfo(operand_index.index(),
                                               createBackwardTensorInfo(operand), ir::Layout::NHWC);
  }

  // TODO Reuse registered tensors when they are planned for memory optimization.
  const auto border = tgraph.getEssentialBackwardOrder();
  for (const auto op_index : border)
  {
    const auto &trainable_op = tgraph.operation(op_index);
    assert(trainable_op.isRequiredForBackward());
    // This assumes that back-propagated tensors of loss outputs are not used
    for (const auto &ind :
         trainable_op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      if (tensor_builder->isRegisteredBackward(ind))
        continue;
      if (external_operands().contains(ind))
        continue;

      const auto &training_usedefs = tgraph.trainingUseDefs();
      const auto &usedefs = training_usedefs.at(ir::train::TrainingOperandIndex{ind, false});
      const bool not_used = usedefs.getTrainingDefs().empty() && usedefs.getTrainingUses().empty();
      if (not_used)
        continue;

      // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
      assert(tgraph.layout() != ir::Layout::NCHW);

      const auto &operand = tgraph.operands().at(ind);
      tensor_builder->registerBackwardTensorInfo(ind, createBackwardTensorInfo(operand),
                                                 ir::Layout::NHWC);
    }
  }

  // Plan tensors only in backwarding to reduce peak memory usage
  planGradientTensors(*this);
  planBackPropTensors(*this);

  for (const auto &op_index : tgraph.getEssentialBackwardOrder())
  {
    const auto back_prop_seq = getBackPropSeq(tgraph, op_index);
    for (const auto &back_prop_index : back_prop_seq)
    {
      DisposableTensorIndex disposable_index{op_index, back_prop_index};
      const auto &operand = tgraph.operands().at(back_prop_index);
      tensor_builder->registerDisposableBackwardTensorInfo(
        disposable_index, createBackwardTensorInfo(operand), ir::Layout::NHWC);
    }
  }

  planDisposableBackPropTensors();

  tensor_builder->allocateBackward();

  return _tensor_registry.get();
}

void BackendContext::planDisposableBackPropTensors()
{
  VERBOSE(BackendContext) << "Start planning disposable back-prop tensors" << std::endl;
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;

  std::vector<DisposableTensorIndex> prev_seq;
  for (const auto &op_index : tgraph.getEssentialBackwardOrder())
  {
    for (const auto &index : prev_seq)
    {
      tensor_builder->notifyDisposableBackPropLastUse(index);
    }

    std::vector<DisposableTensorIndex> cur_seq;
    const auto back_prop_indices = getBackPropSeq(tgraph, op_index);
    for (const auto &back_prop_index : back_prop_indices)
    {
      DisposableTensorIndex cur_index{op_index, back_prop_index};
      tensor_builder->notifyDisposableBackPropFirstUse(cur_index);

      cur_seq.emplace_back(cur_index);
    }

    prev_seq = cur_seq;
  }

  VERBOSE(BackendContext) << "Finish planning disposable back-prop tensors" << std::endl;
}

FunctionMap BackendContext::genKernels()
{
  auto ret = generateFunctionMap();

  // Initialize TrainableTensors
  trainable_graph()->operands().iterate(
    [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      if (external_operands().contains(ind) || !operand.isConstant())
        return;

      auto tensor = tensor_registry()->getNativeITensor(ind);
      assert(tensor != nullptr);

      VERBOSE(FillOperandData) << "Fill data for " << ind << std::endl;

      auto data = operand.shareData();
      assert(data && data->base());
      auto trainable_tensor = dynamic_cast<TrainableTensor *>(tensor);

      if (trainable_tensor == nullptr)
        throw std::runtime_error{"This tensor is not trainable tensor"};

      trainable_tensor->fillBuffer(data);
    });

  // NOTE For memory optimization, we want to free some operand data
  const_cast<ir::train::TrainableGraph &>(*_tdata->tgraph)
    .operands()
    .iterate([&](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

  // TODO Enable
  // for (auto &&it : ret)
  // {
  //   auto &fn_seq = it.second;
  //   fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  // }

  return ret;
}

FunctionMap BackendContext::generateFunctionMap()
{
  train::FunctionMap ret;

  for (const auto &op_ind : _tdata->op_order)
  {
    auto fn_seq = kernel_gen->generate(op_ind);
    ret.emplace(op_ind, std::move(fn_seq));
  }

  // NOTE Each BackPropInitializer should be called first in each op node during backwarding
  const auto &tgraph = *_tdata->tgraph;
  auto tensor_reg = nnfw::misc::polymorphic_downcast<TensorRegistry *>(_tensor_registry.get());
  AddBackPropInitializers(tgraph, *tensor_reg, ret);

  return ret;
}

} // namespace train
} // namespace backend
} // namespace onert
