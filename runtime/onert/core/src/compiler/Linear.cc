/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <algorithm>

#include "Linear.h"

#include "backend/IConfig.h"
#include "backend/IConstantInitializer.h"
#include "backend/ITensorRegister.h"
#include "backend/Backend.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{

std::vector<ir::OpSequenceIndex> Linear::linearize(const ir::LoweredGraph &lowered_graph)
{
  std::vector<ir::OpSequenceIndex> order;
  lowered_graph.iterateTopolOpSeqs(
      [&](const ir::OpSequenceIndex &index, const ir::OpSequence &) -> void {
        order.emplace_back(index);
      });
  return order;
}

void Linear::dump(const ir::LoweredGraph &lowered_graph,
                  const std::vector<ir::OpSequenceIndex> &order)
{
  {
    const auto &toString = [](const onert::backend::Backend *backend) {
      assert(backend);
      std::string str;
      str += backend->config()->id();
      return "{" + str + "}";
    };

    VERBOSE(Linear) << "Final OpSequence" << std::endl;
    for (const auto index : order)
    {
      const auto &op_seq = lowered_graph.op_seqs().at(index);
      const auto lower_info = lowered_graph.getLowerInfo(index);
      const auto &operations = lowered_graph.graph().operations();
      VERBOSE(Linear) << "* OP_SEQ " << toString(lower_info->backend()) << " "
                      << ir::getStrFromOpSeq(op_seq, operations) << std::endl;
    }
  }
}

void Linear::planTensors(const ir::LoweredGraph &lowered_graph,
                         const std::vector<ir::OpSequenceIndex> &order)
{
  const auto &graph = lowered_graph.graph();
  ir::OperandIndexMap<std::shared_ptr<backend::ITensorBuilder>> tensor_builder_map;

  ir::OperandIndexMap<uint32_t> uses_map;
  ir::OperandIndexMap<uint32_t> def_map;
  ir::OperandIndexSequence constants;

  // Prepare scanning
  graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    const auto lower_info = lowered_graph.getLowerInfo(ind);
    // TODO Remove if onert doesn't support anymore such as
    // GeneratedTests.reshape_quant8_weights_as_inputs
    if (lower_info->def_factors().size() == 0 && lower_info->use_factors().size() == 0 &&
        !graph.getInputs().contains(ind))
    {
      VERBOSE(LINEAR) << "Operand #" << ind.value() << " will not be used. no more process."
                      << std::endl;
      return;
    }

    // Unused input of subgraph
    // TODO Register unused input as nullptr in tensor_builder
    if (lower_info->def_factors().size() == 0 && lower_info->use_factors().size() == 0 &&
        graph.getInputs().contains(ind))
    {
      VERBOSE(LINEAR) << "Operand #" << ind.value() << " will not be used. no more process."
                      << std::endl;
      return;
    }

    uses_map[ind] = obj.getUses().size();
    def_map[ind] = obj.getDef().valid() ? 1 : 0;

    bool is_const = obj.isConstant();
    if (is_const)
    {
      constants.append(ind);
    }

    auto factor = lower_info->def_factors().getOnlyElement();
    auto backend = factor.backend();
    auto tensor_builder = lowered_graph.backend_contexts().at(backend)->tensor_builder;
    if (!tensor_builder->isRegistered(ind))
    {
      // These tensors do not exist in any op_seq (No use and def)
      const auto info = obj.info();
      const auto backend_layout = factor.layout();
      // TODO Change tensor info to have permuted shape
      tensor_builder->registerTensorInfo(ind, info, backend_layout);
    }

    tensor_builder_map[ind] = tensor_builder;
  });

  // If a tensor is model output, increase the use of the tensor.
  // This aim is same to above one.
  for (const auto &ind : graph.getOutputs() | ir::Remove::DUPLICATED)
  {
    uses_map[ind]++;
  }

  // Start scanning to do notify{First|Last}Use for each tensor

  // If a tensor is a constant, increase the use of the tensor.
  // It makes the tensor not be dealloced. It means these will be deallocated last.
  // And allocate constant operands first
  VERBOSE(LINEAR) << "TENSORS as CONSTANT" << std::endl;
  for (const auto &ind : constants)
  {
    uses_map[ind]++;
    tensor_builder_map[ind]->notifyFirstUse(ind);
  }

  // Allocate Model's inputs
  VERBOSE(LINEAR) << "TENSORS as MODEL INPUT" << std::endl;
  for (const auto &ind : graph.getInputs() | ir::Remove::DUPLICATED)
  {
    auto tensor_builder = tensor_builder_map[ind];
    if (!tensor_builder) // for GeneratedTests.xxx_weights_as_inputs
      continue;
    tensor_builder->notifyFirstUse(ind);
  }

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  VERBOSE(LINEAR) << "TENSORS" << std::endl;
  for (const auto op_seq_ind : order)
  {
    const auto &op_seq = lowered_graph.op_seqs().at(op_seq_ind);
    for (const auto &op_idx : op_seq.operations())
    {
      for (const auto &ind : graph.operations().at(op_idx).getOutputs() | ir::Remove::DUPLICATED |
                                 ir::Remove::UNDEFINED)
      {
        assert(def_map.find(ind) != def_map.end());
        if (def_map[ind])
        {
          def_map[ind] = 0;
          tensor_builder_map[ind]->notifyFirstUse(ind);
        }
      }

      for (const auto &ind : graph.operations().at(op_idx).getInputs() | ir::Remove::DUPLICATED |
                                 ir::Remove::UNDEFINED)
      {
        assert(uses_map.find(ind) != uses_map.end());
        assert(uses_map[ind] > 0);
        uses_map[ind]--;
        if (uses_map[ind] == 0)
        {
          // plan for deallocation of static tensornode
          tensor_builder_map[ind]->notifyLastUse(ind);

          // plan for deallocation of dynamic tensor
          auto dyn_tensor_manager = tensor_builder_map[ind]->dynamicTensorManager();
          if (dyn_tensor_manager)
            dyn_tensor_manager->planDealloc(op_idx, ind);
        }
      }
    }
  }

  // Dispose and validate
  for (const auto &ind : graph.getOutputs() | ir::Remove::DUPLICATED)
  {
    --uses_map[ind];
    if (uses_map[ind] == 0) // To prevent notifyLastUse from being called twice
    {
      tensor_builder_map[ind]->notifyLastUse(ind);
    }
  }

  for (const auto &ind : constants)
  {
    --uses_map[ind];
    if (uses_map[ind] == 0) // To prevent notifyLastUse from being called twice
    {
      tensor_builder_map[ind]->notifyLastUse(ind);
    }
  }

  assert(
      std::all_of(uses_map.begin(), uses_map.end(),
                  [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));

  assert(
      std::all_of(def_map.begin(), def_map.end(),
                  [](std::pair<const ir::OperandIndex, uint32_t> it) { return it.second == 0; }));
}

} // namespace compiler
} // namespace onert
