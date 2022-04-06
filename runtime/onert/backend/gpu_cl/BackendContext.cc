/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConstantInitializer.h"
#include "TensorBuilder.h"
#include "KernelGenerator.h"

#include "util/logging.h"
#include "ir/Index.h"
#include "ir/Operations.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperandIndexSequence.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

void BackendContext::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                        ir::Layout backend_layout)
{
  TensorType type = TensorType::TENSOR_TYPE_VALID;
  tensor_builder->registerTensorInfo(ind, info, backend_layout, type);
}

ITensorRegistry *BackendContext::genTensors()
{
  ir::OperandIndexMap<TensorType> type_map;

  for (const auto &ind : graph()->getInputs())
  {
    type_map[ind] = TensorType::TENSOR_TYPE_INPUT;
  }

  for (const auto &ind : graph()->getOutputs())
  {
    type_map[ind] = TensorType::TENSOR_TYPE_OUTPUT;
  }
  graph()->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands().contains(ind))
      return;

    const auto frontend_layout = graph()->layout();
    const auto backend_layout = operand_layouts().at(ind);
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    if (obj.isConstant())
    {
      type_map[ind] = TensorType::TENSOR_TYPE_INPUT;
    }
    tensor_builder->registerTensorInfo(ind, backend_info, backend_layout, type_map[ind]);
  });

  // TODO Get compiler options from compiler, and use it rather than getting it from Env
  if (util::getConfigString(util::config::EXECUTOR) == "Linear")
  {
    planTensors();
  }
  else
  {
    // For the executors that does not have fixed linear execution order:
    // To make tensors never be deallocated, this is a workaround to use static memory planner
    graph()->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
      if (tensor_builder->isRegistered(ind))
        tensor_builder->notifyFirstUse(ind);
    });
  }
  tensor_builder->prepare();
  return tensor_registry.get();
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
