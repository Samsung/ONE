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

#include <backend/basic/train/TrainableBackendContextHelpers.h>
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace backend
{
namespace train
{

backend::ITensorRegistry *BackendContext::genTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;
  auto tensor_reg = _tensor_registry;

  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands().contains(ind))
      return;
    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);

    ir::OperandInfo backend_info{obj.shape(), obj.typeInfo(), obj.info().memAllocType(),
                                 obj.isConstant()};
    tensor_builder->registerForwardTensorInfo(ind, backend_info, ir::Layout::NHWC);
  });

  // TODO Plan tensor builds to reduce peak memory usage
  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (tensor_builder->isRegistered(ind))
      tensor_builder->notifyForwardFirstUse(ind);
  });

  tensor_builder->allocateForwardTensors();

  return _tensor_registry.get();
}

backend::train::ITensorRegistry *BackendContext::genTrainingTensors()
{
  genBackwardTensors();

  return _tensor_registry.get();
}

void BackendContext::genBackwardTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _tensor_builder;
  auto tensor_reg = _tensor_registry;

  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands().contains(ind))
      return;
    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);

    if (obj.isConstant())
    {
      ir::OperandInfo backend_info{obj.shape(), obj.typeInfo(), obj.info().memAllocType(), true};
      tensor_builder->registerBackwardTensorInfo(ind, backend_info, ir::Layout::NHWC);
    }
    else
    {
      // For derivative tensor
      assert(tgraph.derivatives().exist(ind));
      const auto &deriv = tgraph.derivatives().at(ind);
      ir::OperandInfo backend_info{deriv.shape(), deriv.typeInfo(), deriv.info().memAllocType(),
                                   false};

      tensor_builder->registerBackwardTensorInfo(ind, backend_info, ir::Layout::NHWC);
    }
  });

  // TODO Plan tensor builds to reduce peak memory usage
  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (tensor_builder->isRegistered(ind))
      tensor_builder->notifyBackwardFirstUse(ind);
  });

  tensor_builder->allocateBackwardTensors();
}

FunctionMap BackendContext::genKernels()
{
  train::FunctionMap ret;

  for (const auto &op_ind : _tdata->op_order)
  {
    auto fn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(fn_seq));
  }

  // Initialize TrainableTensors
  trainable_graph()->operands().iterate(
    [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      // NOTE For now, whether or not to build operands to trainable tensors depends on whether
      //      the corresponding operand is constant.
      if (external_operands().contains(ind) || !operand.isConstant())
        return;

      auto tensor = tensor_registry()->getNativeITensor(ind);
      assert(tensor != nullptr);

      VERBOSE(FillOperandData) << "Fill data for " << ind << "into a trainable tensor" << std::endl;

      auto data = operand.shareData();
      assert(data && data->base());
      auto trainable_tensor = nnfw::misc::polymorphic_downcast<TrainableTensor *>(tensor);

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

} // namespace train
} // namespace backend
} // namespace onert
