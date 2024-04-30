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
void AddBackPropInitializers(const ir::train::TrainableGraph &tgraph, TensorRegistry &tensor_reg,
                             FunctionMap &fn_map)
{
  util::Set<ir::OperandIndex> unvisited;
  tgraph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &operand) {
    // TODO Consider not adding BackPropInitializer if the coresponding BackPropTensors don't
    //      require initilization (i.g. BackPropTensors that are not back-propagated)
    if (!tgraph.getInputs().contains(index) && !operand.isConstant())
      unvisited.add(index);
  });

  for (const auto &op_index : tgraph.btopolSortOperations())
  {
    assert(fn_map.find(op_index) != fn_map.end());

    auto &tn_seq = fn_map.at(op_index);

    // The function added lastest is executed first in a sequence during backwarding.
    std::vector<BackPropTensor *> back_props;
    const auto &op = tgraph.operations().at(op_index);
    for (const auto &back_prop_index :
         op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
    {
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
} // namespace

backend::ITensorRegistry *BackendContext::genTensors()
{
  return basic::train::genTensors(*this, _tensor_builder);
}

backend::train::ITensorRegistry *BackendContext::genTrainingTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;
  tgraph.operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &op) {
    const auto trainable_op = dynamic_cast<const ir::train::TrainableOperation *>(&op);
    assert(trainable_op);
    if (!trainable_op->isRequiredForBackward())
    {
      return;
    }
    for (auto &&ind :
         (op.getInputs() + op.getOutputs()) | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      if (tensor_builder->isRegisteredBackward(ind))
        continue;
      if (external_operands().contains(ind))
        continue;
      // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
      assert(tgraph.layout() != ir::Layout::NCHW);

      const auto &operand = tgraph.operands().at(ind);
      // TODO Different shape of back propagation tensor
      ir::OperandInfo backend_info{operand.shape(), operand.typeInfo(),
                                   operand.info().memAllocType(), operand.isConstant()};
      tensor_builder->registerBackwardTensorInfo(ind, backend_info, ir::Layout::NHWC);
    }
  });

  // TODO Plan tensor builds to reduce peak memory usage
  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (tensor_builder->isRegisteredBackward(ind))
      tensor_builder->notifyBackwardFirstUse(ind);
  });

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
