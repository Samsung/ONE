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

#ifndef __ONERT_BACKEND_BASIC_BACKEND_CONTEXT_HELPERS_H__
#define __ONERT_BACKEND_BASIC_BACKEND_CONTEXT_HELPERS_H__

#include "Tensor.h"
#include "backend/BackendContext.h"
#include "backend/ITensorRegistry.h"
#include "backend/basic/TensorRegistry.h"
#include "compiler/GraphLowerInfo.h"
#include "ir/Index.h"
#include "util/logging.h"

#include <vector>

namespace onert::backend::basic
{

// TODO Remove the template param BackendContext once unification of cpu backend context is done
template <typename T_TensorBuilder>
void planTensors(const std::shared_ptr<T_TensorBuilder> &tensor_builder, const ir::Graph &graph,
                 const util::Set<ir::OperandIndex> &external_operands,
                 const std::vector<onert::ir::OperationIndex> &op_order)
{
  ir::OperandIndexMap<uint32_t> uses_map;
  ir::OperandIndexMap<uint32_t> def_map;
  ir::OperandIndexSequence constants;

  // Prepare scanning
  graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands.contains(ind))
      return;

    // TODO Check if we need to handle unused tensors

    uses_map[ind] = obj.getUses().size();
    def_map[ind] = obj.getDef().valid() ? 1 : 0;

    if (obj.isConstant())
      constants.append(ind);

    if (!tensor_builder->isRegistered(ind))
    {
      // These tensors do not exist in any  (No use and def)
      const auto &info = obj.info();
      tensor_builder->registerTensorInfo(ind, info);
    }
  });

  // Start scanning to do notify{First|Last}Use for each tensor

  // If a tensor is a constant, increase the use of the tensor and allocate it first.
  // Increasing use count here makes the tensor never be deallocated, i.e it they will be
  // deallocated last.
  for (const auto &ind : constants)
  {
    uses_map[ind]++;
    tensor_builder->notifyFirstUse(ind);
  }

  for (const auto &[ind, def_count] : def_map)
  {
    if (def_count == 0)
      tensor_builder->notifyFirstUse(ind);
  }

  // This is a workaround to keep the operands over the execution
  // (the operands look like they are unused)
  std::vector<ir::OperandIndex> operands_last_until_end;
  for (const auto &[ind, use_count] : uses_map)
  {
    if (use_count == 0)
      operands_last_until_end.push_back(ind);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, allocate it
  // 3. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  for (const auto &op_ind : op_order)
  {
    const auto &op = graph.operations().at(op_ind);
    auto op_inputs = op.getUsedInputSet();
    auto op_outputs = op.getUsedOutputSet();

    // Define outputs
    for (const auto &ind : op_outputs)
    {
      if (external_operands.contains(ind))
        continue;
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
      if (external_operands.contains(ind))
        continue;
      if (!tensor_builder->isRegistered(ind))
        continue;
      const auto &operand = graph.operands().at(ind);
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
      if (external_operands.contains(ind))
        continue;
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

  for (const auto &ind : operands_last_until_end)
  {
    tensor_builder->notifyLastUse(ind);
  }

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

template <typename T_TensorBuilder>
util::Set<ir::OperandIndex> register_source_memory_tensors(
  const std::shared_ptr<T_TensorBuilder> &tensor_builder, const ir::Graph &graph,
  const util::Set<ir::OperandIndex> &external_operands,
  const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operand_idx)
{
  // process source tensors that share memory at first
  util::Set<ir::OperandIndex> registered_source_ind;
  for (const auto &[_, source_ind] : shared_memory_operand_idx)
  {
    if (external_operands.contains(source_ind))
      continue;
    if (tensor_builder->isRegistered(source_ind)) // some tensors can have the same source
      continue;
    tensor_builder->registerTensorInfo(source_ind, graph.operands().at(source_ind).info());
    registered_source_ind.add(source_ind);
  }
  return registered_source_ind;
}

template <typename T_TensorBuilder>
ITensorRegistry *genTensors(const std::shared_ptr<T_TensorBuilder> &tensor_builder,
                            const ir::Graph &graph,
                            const util::Set<ir::OperandIndex> &external_operands,
                            const std::shared_ptr<ITensorRegistry> &tensor_registry,
                            const std::vector<onert::ir::OperationIndex> &op_order,
                            const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operand_idx)
{
  const auto registered_source_ind = register_source_memory_tensors(
    tensor_builder, graph, external_operands, shared_memory_operand_idx);
  graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (external_operands.contains(ind))
      return;
    if (registered_source_ind.contains(ind)) // skip tensors already registered
      return;
    tensor_builder->registerTensorInfo(ind, obj.info());
  });

  // TODO Get compiler options from compiler, and use it rather than getting it from Env
  if (util::getConfigString(util::config::EXECUTOR) == "Linear")
  {
    basic::planTensors(tensor_builder, graph, external_operands, op_order);
  }
  else
  {
    // For the executors that does not have fixed linear execution order:
    // To make tensors never be deallocated, this is a workaround to use static memory planner
    graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
      if (tensor_builder->isRegistered(ind))
        tensor_builder->notifyFirstUse(ind);
    });
  }

  tensor_builder->allocate();

  return tensor_registry.get();
}

template <typename T_BackendContext> ITensorRegistry *genTensors(T_BackendContext &ctx)
{
  return genTensors(ctx.tensor_builder, *ctx.graph(), ctx.external_operands(), ctx.tensor_registry,
                    ctx.data().op_order, {});
}

inline void
initSharedMemoryConsts(const ir::Operands &operands,
                       const util::Set<ir::OperandIndex> &external_operands,
                       TensorRegistry *tensor_registry,
                       const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operands_map)
{
  operands.iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (external_operands.contains(ind))
      return;
    const auto shared_mem_it = shared_memory_operands_map.find(ind);
    if (shared_mem_it == std::end(shared_memory_operands_map))
      return; // no shared memory source
    if (!operands.at(shared_mem_it->second).isConstant())
      return; // source operand not a constant

    VERBOSE(FillOperandData) << "Fill shared data for " << ind << std::endl;

    const auto &source_operand_ind = operands.at(shared_mem_it->second);
    auto memory_source_data = source_operand_ind.shareData();
    assert(memory_source_data && memory_source_data->base());
    auto tensor = tensor_registry->getNativeTensor(ind);
    assert(tensor != nullptr);
    tensor->setBuffer(const_cast<uint8_t *>(memory_source_data->base()));
  });
}

inline void initConsts(const ir::Operands &operands,
                       const util::Set<ir::OperandIndex> &external_operands,
                       ITensorRegistry *tensor_registry,
                       const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operands_map)
{
  operands.iterate([&](const ir::OperandIndex &ind, const ir::Operand &operand) {
    if (external_operands.contains(ind) || !operand.isConstant())
      return;
    const bool has_const_shared_source =
      shared_memory_operands_map.find(ind) != std::end(shared_memory_operands_map) &&
      operands.at(shared_memory_operands_map.at(ind)).isConstant();
    if (has_const_shared_source)
      return; // tensors with shared memory are processed in initSharedMemoryConsts

    auto tensor = tensor_registry->getNativeITensor(ind);
    assert(tensor != nullptr);

    VERBOSE(FillOperandData) << "Fill data for " << ind << std::endl;

    auto data = operand.shareData();
    assert(data && data->base());
    ExternalTensor *ext_tensor = dynamic_cast<ExternalTensor *>(tensor);

    if (ext_tensor == nullptr)
      throw std::runtime_error{"This tensor is not external tensor"};

    ext_tensor->setData(data);
  });
}

inline void initConsts(BackendContext &ctx)
{
  initConsts(ctx.graph()->operands(), ctx.external_operands(), ctx.tensor_registry.get(), {});
}

} // namespace onert::backend::basic

#endif // __ONERT_BACKEND_BASIC_BACKEND_CONTEXT_HELPERS_H__
