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

#include <backend/basic/BackendContextHelpers.h>

namespace onert
{
namespace backend
{
namespace train
{

ITensorRegistry *BackendContext::genTensors() { return basic::genTensors(*this, _tensor_builder); }

ITensorRegistry *BackendContext::genTrainingTensors()
{
  genGradTensors();

  // TODO Generate training-related tensors except for gradient

  return _grad_tensor_registry.get();
}

void BackendContext::genGradTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _grad_tensor_builder;
  auto tensor_reg = _grad_tensor_registry;

  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (external_operands().contains(ind))
      return;
    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);

    // TODO Register TensorInfo that has gradient's shape
    // ir::OperandInfo backend_info{obj.shape(), obj.typeInfo(), obj.info().memAllocType(),
    //                              obj.isConstant()};
    // tensor_builder->registerTensorInfo(ind, backend_info, ir::Layout::NHWC);
  });

  // TODO Plan tensor builds to reduce peak memory usage
  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (tensor_builder->isRegistered(ind))
      tensor_builder->notifyFirstUse(ind);
  });

  // TODO Allocate tensors
  // tensor_builder->allocate();
}

FunctionMap BackendContext::genKernels()
{
  train::FunctionMap ret;

  // TODO Generate TrainableSeqeunce
  for (auto op_ind : _tdata->op_order)
  {
    auto fn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(fn_seq));
  }

  basic::initConsts(*this);

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
