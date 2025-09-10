/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "util/logging.h"
#include "ir/Index.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperandIndexSequence.h"
#include "backend/basic/BackendContextHelpers.h"
#include "backend/basic/TensorRegistry.h"

#include <misc/polymorphic_downcast.h>

namespace onert::backend::cpu
{

ITensorRegistry *BackendContext::genTensors()
{
  return basic::genTensors(_tensor_builder, *graph(), external_operands(), _tensor_registry,
                           data().op_order, _tensor_builder->getSharedMemoryOperandIndexes());
}

FunctionMap BackendContext::genKernels()
{
  FunctionMap ret;

  basic::initConsts(graph()->operands(), external_operands(), _tensor_registry.get(),
                    _tensor_builder->getSharedMemoryOperandIndexes());

  // TODO: Change type of tensor_registry field to TensorRegistry
  auto tensor_registry_concreted =
    nnfw::misc::polymorphic_downcast<basic::TensorRegistry *>(_tensor_registry.get());
  basic::initSharedMemoryConsts(graph()->operands(), external_operands(), tensor_registry_concreted,
                                _tensor_builder->getSharedMemoryOperandIndexes());

  for (auto &&op_ind : _data.op_order)
  {
    auto fn_seq = _kernel_gen->generate(op_ind);
    ret.emplace(op_ind, std::move(fn_seq));
  }

  // NOTE For memory optimization, we want to free some operand data
  const_cast<ir::Graph &>(*_data.graph)
    .operands()
    .iterate([&](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

  for (auto &&it : ret)
  {
    auto &fn_seq = it.second;
    fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  }

  return ret;
}

} // namespace onert::backend::cpu
