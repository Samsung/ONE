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
#include "TensorPlanner.h"
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

util::Set<ir::train::TrainingOperandIndex>
getBackwardTensorList(const ir::train::TrainableGraph &tgraph,
                      const util::Set<ir::OperandIndex> &external_operands)
{
  util::Set<ir::train::TrainingOperandIndex> ret;

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
      if (external_operands.contains(ind))
        continue;

      const auto &operand_index = ir::train::TrainingOperandIndex{ind, false};

      const auto &training_usedefs = tgraph.trainingUseDefs();
      const auto &usedefs = training_usedefs.at(ir::train::TrainingOperandIndex{ind, false});
      const bool not_used = usedefs.getTrainingDefs().empty() && usedefs.getTrainingUses().empty();
      if (not_used)
        continue;

      ret.add(operand_index);
    }
  }

  return ret;
}

util::Set<DisposableTensorIndex>
getDisposableBackPropTensorList(const ir::train::TrainableGraph &tgraph,
                                const util::Set<ir::OperandIndex> &external_operands)
{
  util::Set<DisposableTensorIndex> ret;

  const auto candidates = getBackwardTensorList(tgraph, external_operands);
  for (const auto &backwarding_operand_index : candidates)
  {
    const auto &operand = tgraph.operands().at(backwarding_operand_index.index());
    const auto &training_usedefs = tgraph.trainingUseDefs();
    const auto &usedefs = training_usedefs.at(backwarding_operand_index);
    // if (!operand.isConstant())
    const bool is_multiple_defs = usedefs.getTrainingDefs().size() > 1;
    if (!operand.isConstant() && is_multiple_defs)
      for (const auto &def : usedefs.getTrainingDefs())
        ret.add(DisposableTensorIndex{def.index(), backwarding_operand_index.index()});
  }

  return ret;
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

  const auto ctx_data = data();
  TensorPlanner tensor_planner{*ctx_data->tgraph.get(), ctx_data->external_operands};
  tensor_planner.planTrainableTensors(_tensor_builder.get());
  tensor_planner.planNonConstTensors(_tensor_builder.get());

  _tensor_builder->allocate();

  return _tensor_registry.get();
}

backend::train::ITensorRegistry *BackendContext::genTrainingTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;

  const auto operand_indices = getBackwardTensorList(tgraph, external_operands());
  for (const auto &operand_index : operand_indices)
  {
    if (external_operands().contains(operand_index.index()))
      continue;

    assert(operand_index.valid());

    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);
    assert(!operand_index.is_forward());
    const auto &operand = tgraph.operands().at(operand_index.index());
    tensor_builder->registerBackwardTensorInfo(operand_index.index(),
                                               createBackwardTensorInfo(operand), ir::Layout::NHWC);
  }

  const auto disposable_indices = getDisposableBackPropTensorList(tgraph, external_operands());
  for (const auto &disposable_index : disposable_indices)
  {
    const auto &operand = tgraph.operands().at(disposable_index.operand_index());
    tensor_builder->registerDisposableBackwardTensorInfo(
      disposable_index, createBackwardTensorInfo(operand), ir::Layout::NHWC);
  }

  // // TODO Register disposable tensor info only when it is necessary
  // const auto border = tgraph.getEssentialBackwardOrder();
  // for (const auto &op_index : border)
  // {
  //   // const auto back_prop_seq = getBackPropSeq(tgraph, op_index);
  //   const auto &trainable_op = tgraph.operation(op_index);
  //   const auto back_prop_seq =
  //     trainable_op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
  //   for (const auto &input : back_prop_seq)
  //   {
  //     if (external_operands().contains(input))
  //       continue;

  //     const auto backwarding_index = ir::train::TrainingOperandIndex{input, false};
  //     const auto &training_usedefs = tgraph.trainingUseDefs();
  //     const auto &usedefs = training_usedefs.at(backwarding_index);
  //     const bool not_used = usedefs.getTrainingDefs().empty() &&
  //     usedefs.getTrainingUses().empty(); if (not_used)
  //       continue;

  //     const auto &operand = tgraph.operands().at(input);
  //     if (!operand.isConstant())
  //     {
  //       // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
  //       assert(tgraph.layout() != ir::Layout::NCHW);
  //       DisposableTensorIndex disposable_index{op_index, input};
  //       tensor_builder->registerDisposableBackwardTensorInfo(
  //         disposable_index, createBackwardTensorInfo(operand), ir::Layout::NHWC);
  //     }
  //   }
  // }

  // Plan tensors only in backwarding to reduce peak memory usage
  const auto ctx_data = data();
  TensorPlanner tensor_planner{*ctx_data->tgraph.get(), ctx_data->external_operands};
  tensor_planner.planGradientTensors(tensor_builder.get());
  tensor_planner.planBackPropTensors(tensor_builder.get());
  tensor_planner.planDisposableBackPropTensors(tensor_builder.get());

  tensor_builder->allocateBackward();

  return _tensor_registry.get();
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
