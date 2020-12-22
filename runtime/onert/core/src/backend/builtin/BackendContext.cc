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

#include "KernelGenerator.h"
#include "backend/cpu_common/BackendContextHelpers.h"

namespace onert
{
namespace backend
{
namespace builtin
{

ITensorRegistry *BackendContext::genTensors(const std::vector<onert::ir::OpSequenceIndex> &order,
                                            const ir::OpSequences &op_seqs,
                                            const ir::LowerInfoMap &lower_info)
{
  auto model_io =
    (graph()->getInputs() + graph()->getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  for (auto index : operand_list())
  {
    if (model_io.contains(index))
      continue;
    const auto &obj = graph()->operands().at(index);
    const auto frontend_layout = [&]() {
      if (obj.getUses().size() == 0)
        return ir::Layout::UNKNOWN;
      auto use_op_ind = *obj.getUses().begin(); // FIXME What if it has two or more uses?
      for (auto &operation_info : operation_list())
      {
        if (operation_info.index == use_op_ind)
          return operation_info.layout;
      }
      return ir::Layout::UNKNOWN;
    }();
    const auto &permute_factor = lower_info.operand.at(index)->def_factors().getOnlyElement();
    if (permute_factor.backend() != backend())
      continue;
    const auto backend_layout = permute_factor.layout();
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    tensor_builder->registerTensorInfo(index, backend_info, backend_layout);
  }

  // TODO Get compiler options from compiler, and use it rather than getting it from Env
  if (util::getConfigString(util::config::EXECUTOR) == "Linear")
  {
    cpu_common::planTensors(*this, order, op_seqs, lower_info);
  }
  else
  {
    // For the executors that does not have fixed linear execution order:
    // To make tensors never be deallocated, this is a workaround to use static memory planner
    for (auto ind : operand_list())
    {
      if (tensor_builder->isRegistered(ind))
        tensor_builder->notifyFirstUse(ind);
    }
  }

  tensor_builder->allocate();

  return tensor_registry.get();
}

FunctionMap BackendContext::genKernels(const std::vector<ir::OpSequenceIndex> &order,
                                       const ir::OpSequences &op_seqs)
{
  FunctionMap ret;

  for (auto op_seq_ind : order)
  {
    const auto &op_seq = op_seqs.at(op_seq_ind);
    bool assigned = [&]() {
      for (auto op_info : operation_list())
        if (op_seq.exist(op_info.index))
          return true;
      return false;
    }();
    if (!assigned)
      continue;
    auto fn_seq = kernel_gen->generate(op_seqs.at(op_seq_ind));
    ret.emplace_back(op_seq_ind, std::move(fn_seq));
  }

  cpu_common::initConsts(*this);

  // NOTE For memory optimization, we want to free some operand data
  for (auto ind : operand_list())
  {
    // TODO Remove const_cast
    auto &obj = const_cast<ir::Graph *>(graph())->operands().at(ind);
    obj.releaseData();
  }

  for (auto &it : ret)
  {
    auto &fn_seq = it.second;
    fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  }

  return ret;
}

} // namespace builtin
} // namespace backend
} // namespace onert
