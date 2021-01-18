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
#include "Optimizer.h"
#include "util/logging.h"
#include "ir/Index.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperandIndexSequence.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

void BackendContext::initConsts()
{
  _data.graph->operations().iterate([&](const ir::OperationIndex &ind, const ir::Operation &op) {
    constant_initializer->setLayout(operation_layouts().at(ind));
    op.accept(*constant_initializer);
  });

  _data.graph->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &operand) {
    if (_data.external_operands.contains(ind) || !operand.isConstant())
      return;
    const auto &obj = graph()->operands().at(ind);
    if (obj.isConstant() && !constant_initializer->exist(ind))
    {
      constant_initializer->registerDefaultInitializer(ind, obj);
    }
  });

  constant_initializer->run();
}

void BackendContext::planTensors()
{
  ir::OperandIndexMap<uint32_t> uses_map;
  ir::OperandIndexMap<uint32_t> def_map;
  ir::OperandIndexSequence constants;

  // Prepare scanning
  _data.graph->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (_data.external_operands.contains(ind))
      return;

    uses_map[ind] = obj.getUses().size();
    def_map[ind] = obj.getDef().valid() ? 1 : 0;

    if (obj.isConstant())
      constants.append(ind);

    if (!tensor_builder->isRegistered(ind))
    {
      // These tensors do not exist in any operation (No use and def)
      const auto info = obj.info();
      const auto layout = _data.operand_layouts.at(ind);
      // TODO Change tensor info to have permuted shape
      tensor_builder->registerTensorInfo(ind, info, layout);
    }
  });

  // Start scanning to do notify{First|Last}Use for each tensor

  // If a tensor is a constant, increase the use of the tensor and allocate it first.
  // Increasing use count here makes the tensor never be deallocated, i.e it they will be
  // deallocated last.
  VERBOSE(planTensors) << "TENSORS as CONSTANT" << std::endl;
  for (const auto &ind : constants)
  {
    uses_map[ind]++;
    tensor_builder->notifyFirstUse(ind);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, allocate it
  // 3. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  for (const auto op_ind : _data.op_order)
  {
    // TODO Remove indentation
    {
      auto op_inputs = graph()->operations().at(op_ind).getInputs() | ir::Remove::DUPLICATED |
                       ir::Remove::UNDEFINED;
      auto op_outputs = graph()->operations().at(op_ind).getOutputs() | ir::Remove::DUPLICATED |
                        ir::Remove::UNDEFINED;

      // Define outputs
      for (const auto &ind : op_outputs)
      {
        if (!tensor_builder->isRegistered(ind))
          continue;
        assert(def_map.find(ind) != def_map.end());
        if (def_map[ind])
        {
          def_map[ind] = 0;
          tensor_builder->notifyFirstUse(ind);
        }
      }

      // Scan variable tensors
      // This tensor has features like constant. But OperandInfo and LowerInfo treat them as
      // non-constant because of less memory usage by memory planning in here
      for (const auto &ind : op_inputs)
      {
        if (!tensor_builder->isRegistered(ind))
          continue;
        const auto &operand = graph()->operands().at(ind);
        if (operand.info().isVariable())
        {
          // The variable tensor with buffer is not supported yet
          assert(operand.data() == nullptr);
          assert(operand.getUses().size() == 1 && !operand.getDef().valid());
          assert(uses_map[ind] == 1 && def_map[ind] == 0);
          tensor_builder->notifyFirstUse(ind);
        }
      }

      for (const auto &ind : op_inputs)
      {
        if (!tensor_builder->isRegistered(ind))
          continue;
        assert(uses_map.find(ind) != uses_map.end());
        assert(uses_map[ind] > 0);
        uses_map[ind]--;
        if (uses_map[ind] == 0)
        {
          // plan for deallocation of static tensornode
          tensor_builder->notifyLastUse(ind);
        }
      }
    }
  }

  _data.graph->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (uses_map[ind] == 0)
    {
      tensor_builder->notifyLastUse(ind);
    }
  });

  // Dispose and validate
  for (const auto &ind : constants)
  {
    --uses_map[ind];
    if (uses_map[ind] == 0) // To prevent notifyLastUse from being called twice
    {
      tensor_builder->notifyLastUse(ind);
    }
  }

  assert(
    std::all_of(uses_map.begin(), uses_map.end(),
                [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));

  assert(
    std::all_of(def_map.begin(), def_map.end(),
                [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));
}

ITensorRegistry *BackendContext::genTensors()
{
  optimizer->optimize();

  graph()->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands().contains(ind))
      return;

    const auto frontend_layout = graph()->layout();
    const auto backend_layout = operand_layouts().at(ind);
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    tensor_builder->registerTensorInfo(ind, backend_info, backend_layout);
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

FunctionMap BackendContext::genKernels()
{
  FunctionMap ret;

  for (auto op_ind : _data.op_order)
  {
    auto fn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(fn_seq));
  }

  tensor_builder->allocate();
  initConsts();

  // NOTE For memory optimization, we want to free some operand data
  const_cast<ir::Graph &>(*_data.graph)
    .operands()
    .iterate([&](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

  for (auto &it : ret)
  {
    auto &fn_seq = it.second;
    fn_seq->iterate([&](exec::IFunction &ifunc) {
      ifunc.prepare();
      tensor_builder->postFunctionPrepare();
    });
  }

  return ret;
}

} // namespace acl_cl
} // namespace backend
} // namespace onert
