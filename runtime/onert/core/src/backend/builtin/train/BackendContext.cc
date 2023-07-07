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

#include "backend/basic/train/TrainableBackendContextHelpers.h"
#include "exec/FunctionSequence.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

ITensorRegistry *BackendContext::genTensors()
{
  return basic::train::genTensors(*this, _tensor_builder);
}

ITensorRegistry *BackendContext::genTrainingTensors()
{
  genGradTensors();

  // TODO Generate training-related tensors except for gradient

  return grad_tensor_registry().get();
}

void BackendContext::genGradTensors()
{
  const ir::train::TrainableGraph &tgraph = *trainable_graph();
  auto tensor_builder = _grad_tensor_builder;
  auto tensor_reg = grad_tensor_registry();

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

backend::train::FunctionMap BackendContext::genKernels()
{
  backend::train::FunctionMap ret;

  for (auto &&op_ind : _tdata->op_order)
  {
    auto tn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(tn_seq));
  }

  basic::train::initConsts(*this);

  // TODO Enable prepare()
  // for (auto &&it : ret)
  // {
  //   auto &fn_seq = it.second;
  //   fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  // }

  return ret;
}

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
