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

#ifndef __ONERT_BACKEND_CPU_COMMON_BACKEND_CONTEXT_HELPERS_H__
#define __ONERT_BACKEND_CPU_COMMON_BACKEND_CONTEXT_HELPERS_H__

#include <vector>

#include "ir/Index.h"
#include "ir/OpSequences.h"
#include "compiler/GraphLowerInfo.h"
#include "util/logging.h"
#include "backend/ITensorRegistry.h"
#include "backend/BackendContext.h"
#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

// TODO Remove the template param BackendContext once unification of cpu backend context is done
template <typename T_BackendContext>
void planTensors(const T_BackendContext &ctx, const std::vector<onert::ir::OpSequenceIndex> &order,
                 const ir::OpSequences &op_seqs, const compiler::GraphLowerInfo &lower_info)
{
  auto graph = ctx.graph();
  auto tensor_builder = ctx.tensor_builder;

  ir::OperandIndexMap<uint32_t> uses_map;
  ir::OperandIndexMap<uint32_t> def_map;
  ir::OperandIndexSequence constants;

  auto model_io =
    (graph->getInputs() + graph->getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;

  // Prepare scanning
  for (auto ind : ctx.operand_list())
  {
    if (model_io.contains(ind))
      continue;
    const auto &obj = graph->operands().at(ind);
    const auto &li = lower_info.operand.at(ind);
    if (li->def_factors().getOnlyElement().backend() != ctx.backend())
      continue;

    // Ignore unused tensor
    if (li->def_factors().size() == 0 && li->use_factors().size() == 0)
    {
      VERBOSE_F() << "Operand " << ind << " will not be used. no more process." << std::endl;
      return;
    }

    uses_map[ind] = obj.getUses().size();
    def_map[ind] = obj.getDef().valid() ? 1 : 0;

    if (obj.isConstant())
      constants.append(ind);

    auto factor = li->def_factors().getOnlyElement();
    if (!tensor_builder->isRegistered(ind))
    {
      // These tensors do not exist in any op_seq (No use and def)
      const auto info = obj.info();
      const auto backend_layout = factor.layout();
      // TODO Change tensor info to have permuted shape
      tensor_builder->registerTensorInfo(ind, info, backend_layout);
    }
  }

  // Start scanning to do notify{First|Last}Use for each tensor

  // If a tensor is a constant, increase the use of the tensor and allocate it first.
  // Increasing use count here makes the tensor never be deallocated, i.e it they will be
  // deallocated last.
  for (const auto &ind : constants)
  {
    uses_map[ind]++;
    tensor_builder->notifyFirstUse(ind);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan DEF of inputs. If variable tensor, allocate it
  // 3. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  for (const auto op_seq_ind : order)
  {
    const auto &op_seq = op_seqs.at(op_seq_ind);
    for (const auto &op_idx : op_seq.operations())
    {
      auto op_inputs =
        graph->operations().at(op_idx).getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED;
      auto op_outputs = graph->operations().at(op_idx).getOutputs() | ir::Remove::DUPLICATED |
                        ir::Remove::UNDEFINED;

      // Define outputs
      for (const auto &ind : op_outputs)
      {
        if (model_io.contains(ind))
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
        if (model_io.contains(ind))
          continue;
        if (!tensor_builder->isRegistered(ind))
          continue;
        const auto &operand = graph->operands().at(ind);
        if (operand.info().isVariable())
        {
          // The variable tensor with buffer is not supported yet
          assert(operand.data() == nullptr);
          assert(operand.getUses().size() == 1 && !operand.getDef().valid());
          assert(lower_info.operand.at(ind)->def_factors().size() == 1 &&
                 lower_info.operand.at(ind)->use_factors().size() == 1);
          assert(uses_map[ind] == 1 && def_map[ind] == 0);
          tensor_builder->notifyFirstUse(ind);
        }
      }

      for (const auto &ind : op_inputs)
      {
        if (model_io.contains(ind))
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

          // plan for deallocation of dynamic tensor
          auto dyn_tensor_manager = tensor_builder->dynamicTensorManager();
          auto *tensor = ctx.tensor_registry->getITensor(ind);
          assert(tensor);
          dyn_tensor_manager->planDealloc(op_idx, tensor);
        }
      }
    }
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

template <typename T_BackendContext>
ITensorRegistry *
genTensors(T_BackendContext &ctx, const std::vector<onert::ir::OpSequenceIndex> &order,
           const ir::OpSequences &op_seqs, const compiler::GraphLowerInfo &lower_info)
{
  auto graph = ctx.graph();
  auto tensor_builder = ctx.tensor_builder;

  auto model_io =
    (graph->getInputs() + graph->getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  for (auto index : ctx.operand_list())
  {
    if (model_io.contains(index))
      continue;
    const auto &obj = graph->operands().at(index);
    const auto frontend_layout = [&]() {
      if (obj.getUses().size() == 0)
        return ir::Layout::UNKNOWN;
      auto use_op_ind = *obj.getUses().begin(); // FIXME What if it has two or more uses?
      for (auto &operation_info : ctx.operation_list())
      {
        if (operation_info.index == use_op_ind)
          return operation_info.layout;
      }
      return ir::Layout::UNKNOWN;
    }();
    const auto &permute_factor = lower_info.operand.at(index)->def_factors().getOnlyElement();
    if (permute_factor.backend() != ctx.backend())
      continue;
    const auto backend_layout = permute_factor.layout();
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    tensor_builder->registerTensorInfo(index, backend_info, backend_layout);
  }

  // TODO Get compiler options from compiler, and use it rather than getting it from Env
  if (util::getConfigString(util::config::EXECUTOR) == "Linear")
  {
    cpu_common::planTensors(ctx, order, op_seqs, lower_info);
  }
  else
  {
    // For the executors that does not have fixed linear execution order:
    // To make tensors never be deallocated, this is a workaround to use static memory planner
    for (auto ind : ctx.operand_list())
    {
      if (tensor_builder->isRegistered(ind))
        tensor_builder->notifyFirstUse(ind);
    }
  }

  tensor_builder->allocate();

  return ctx.tensor_registry.get();
}

inline void initConsts(BackendContext &ctx)
{
  for (auto ind : ctx.operand_list())
  {
    const auto &operand = ctx.graph()->operands().at(ind);
    if (!operand.isConstant())
      continue;

    auto tensor = ctx.tensor_registry->getNativeITensor(ind);
    assert(tensor != nullptr);

    VERBOSE(FillOperandData) << "Fill data for " << ind << std::endl;

    auto data = operand.shareData();
    assert(data && data->base());
    ExternalTensor *ext_tensor = dynamic_cast<ExternalTensor *>(tensor);
    assert(ext_tensor);
    ext_tensor->setData(data);
  }
}

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_BACKEND_CONTEXT_HELPERS_H__
