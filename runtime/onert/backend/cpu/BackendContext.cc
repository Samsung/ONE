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

namespace onert
{
namespace backend
{
namespace cpu
{

void BackendContext::planTensors(const std::vector<onert::ir::OpSequenceIndex> &order,
                                 const ir::OpSequences &op_seqs, const ir::LowerInfoMap &lower_info)
{
  ir::OperandIndexMap<uint32_t> uses_map;
  ir::OperandIndexMap<uint32_t> def_map;
  ir::OperandIndexSequence constants;

  // Prepare scanning
  for (auto ind : operand_list())
  {
    const auto &obj = graph()->operands().at(ind);
    const auto &li = lower_info.operand.at(ind);
    if (li->def_factors().getOnlyElement().backend() != backend())
      continue;

    // Ignore unused tensor
    if (li->def_factors().size() == 0 && li->use_factors().size() == 0)
    {
      VERBOSE(LINEAR) << "Operand #" << ind.value() << " will not be used. no more process."
                      << std::endl;
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
  VERBOSE(LINEAR) << "TENSORS as CONSTANT" << std::endl;
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
      auto op_inputs = graph()->operations().at(op_idx).getInputs() | ir::Remove::DUPLICATED |
                       ir::Remove::UNDEFINED;
      auto op_outputs = graph()->operations().at(op_idx).getOutputs() | ir::Remove::DUPLICATED |
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
          assert(lower_info.operand.at(ind)->def_factors().size() == 1 &&
                 lower_info.operand.at(ind)->use_factors().size() == 1);
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

          // plan for deallocation of dynamic tensor
          auto dyn_tensor_manager = tensor_builder->dynamicTensorManager();
          auto *tensor = tensor_registry->getITensor(ind);
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

ITensorRegistry *BackendContext::tensorGen(const std::vector<onert::ir::OpSequenceIndex> &order,
                                           const ir::OpSequences &op_seqs,
                                           const ir::LowerInfoMap &lower_info)
{
  auto model_io = (graph()->getInputs() + graph()->getOutputs()) | ir::Remove::UNDEFINED |
                  ir::Remove::DUPLICATED;
  for (auto index : operand_list())
  {
    if (model_io.contains(index))
      continue;
    const auto &obj = graph()->operands().at(index);
    const auto frontend_layout = [&]() {
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
    VERBOSE_F() << index << std::endl;
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    tensor_builder->registerTensorInfo(index, backend_info, backend_layout);
  }

  // PLAN TENSORS
  planTensors(order, op_seqs, lower_info);

  tensor_builder->prepare();

  return tensor_registry.get();
}

std::vector<std::pair<ir::OpSequenceIndex, std::unique_ptr<exec::FunctionSequence>>>
BackendContext::kernelGen(const std::vector<onert::ir::OpSequenceIndex> &order,
                          const ir::OpSequences &op_seqs)
{
  std::vector<std::pair<ir::OpSequenceIndex, std::unique_ptr<exec::FunctionSequence>>> ret;

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

  initConsts();

  // NOTE For memory optimization, we want to free some operand data
  // TODO Fix it (remove const_cast)
  const_cast<ir::Graph *>(graph())->operands().iterate(
      [](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

  for (auto &it : ret)
  {
    auto &fn_seq = it.second;
    fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  }

  return ret;
}

} // namespace cpu
} // namespace backend
} // namespace onert
