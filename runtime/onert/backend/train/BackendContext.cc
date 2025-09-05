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

#include "KernelGenerator.h"
#include "TensorBuilder.h"
#include "TensorPlanner.h"
#include "ops/BackPropInitializer.h"

#include <backend/basic/train/TrainableBackendContextHelpers.h>
#include <misc/polymorphic_downcast.h>

#include <cassert>

namespace onert::backend::train
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

  for (const auto &op_index : tgraph.essentialBackwardOrder())
  {
    assert(fn_map.find(op_index) != fn_map.end());

    auto &tn_seq = fn_map.at(op_index);

    // The function added latest is executed first in a sequence during backwarding.
    std::vector<BackPropTensor *> back_props;
    const auto &op = tgraph.operation(op_index);
    for (const auto &back_prop_index : op.getUsedInputSet())
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
  auto border = tgraph.essentialBackwardOrder();
  for (const auto op_index : border)
  {
    const auto &trainable_op = tgraph.operation(op_index);
    assert(trainable_op.isRequiredForBackward());
    // This assumes that back-propagated tensors of loss outputs are not used
    for (const auto &ind : trainable_op.getUsedInputSet())
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
    const bool is_multiple_defs = usedefs.getTrainingDefs().size() > 1;
    if (!operand.isConstant() && is_multiple_defs)
      for (const auto &def : usedefs.getTrainingDefs())
        ret.add(DisposableTensorIndex{def.index(), backwarding_operand_index.index()});
  }

  return ret;
}
} // namespace

FunctionMap BackendContext::gen()
{
  planForwardTensors();
  planBackwardTensors();

  _tensor_builder->allocate();
  _tensor_builder->allocateBackward();

  auto fn_map = generateFunctionMap();

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

  // NOTE: Since LayerScopeTensors is defined in each kernel(layer),
  //       It should be planned and allocated after the kernels generated.
  planLayerScopeTensors(fn_map);
  _tensor_builder->allocateLayerScope();

  return fn_map;
}

void BackendContext::planForwardTensors()
{
  const auto &tgraph = *trainable_graph();

  tgraph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &obj) {
    if (external_operands().contains(index))
      return;
    if (!index.valid())
      return;

    _tensor_builder->registerTensorInfo(index, obj.info());
  });

  const auto ctx_data = data();
  TensorPlanner tensor_planner{*ctx_data->tgraph.get(), ctx_data->external_operands};
  tensor_planner.planTrainableTensors(_tensor_builder.get());
  tensor_planner.planNonConstTensors(_tensor_builder.get());
}

void BackendContext::planBackwardTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();

  auto tensor_builder = _tensor_builder;

  const auto operand_indices = getBackwardTensorList(tgraph, external_operands());
  for (const auto &operand_index : operand_indices)
  {
    if (external_operands().contains(operand_index.index()))
      continue;

    assert(operand_index.valid());

    assert(!operand_index.is_forward());
    const auto &operand = tgraph.operands().at(operand_index.index());
    tensor_builder->registerBackwardTensorInfo(operand_index.index(),
                                               createBackwardTensorInfo(operand));
  }

  const auto disposable_indices = getDisposableBackPropTensorList(tgraph, external_operands());
  for (const auto &disposable_index : disposable_indices)
  {
    const auto &operand = tgraph.operands().at(disposable_index.operand_index());
    tensor_builder->registerDisposableBackwardTensorInfo(disposable_index,
                                                         createBackwardTensorInfo(operand));
  }

  // Plan tensors only in backwarding to reduce peak memory usage
  const auto ctx_data = data();
  TensorPlanner tensor_planner{*ctx_data->tgraph.get(), ctx_data->external_operands};
  tensor_planner.planGradientTensors(tensor_builder.get());
  tensor_planner.planBackPropTensors(tensor_builder.get());
  tensor_planner.planDisposableBackPropTensors(tensor_builder.get());
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

void BackendContext::planLayerScopeTensors([[maybe_unused]] const FunctionMap &fn_map)
{
  const auto &ops = trainable_graph()->operations();

  auto register_tensors = [this](const ir::OperationIndex &op_idx,
                                 std::optional<LayerScopeTensors> &&tensors) {
    if (not tensors.has_value())
      return;

    auto ls_tensors = tensors.value();
    for (auto i = 0u; i < ls_tensors.size(); ++i)
    {
      LayerScopeTensorIndex tensor_idx(op_idx, i);
      _tensor_builder->registerLayerScopeTensor(tensor_idx, ls_tensors[i]);

      VERBOSE(BackendContext) << "(idx:" << tensor_idx << ") registered" << std::endl;
    }
    return;
  };

  for (auto &pair : fn_map)
  {
    const auto &op_idx = pair.first;
    auto &fn_seq = pair.second;

    const ir::IOperation *op = &ops.at(op_idx);
    const auto trainable_op = dynamic_cast<const ir::train::TrainableOperation *>(op);
    assert(trainable_op != nullptr);

    if (not trainable_op->isRequiredForBackward())
      continue;

    VERBOSE(BackendContext) << "register layerscope tensor for " << trainable_op->name()
                            << std::endl;

    fn_seq->iterate([&](exec::train::ITrainableFunction &fn) {
      register_tensors(op_idx, (&fn)->registerLayerScopeTensors());
    });
  }

  const auto ctx_data = data();
  TensorPlanner tensor_planner{*ctx_data->tgraph.get(), ctx_data->external_operands};
  tensor_planner.planLayerScopeTensors(_tensor_builder.get());
  return;
}

} // namespace onert::backend::train
