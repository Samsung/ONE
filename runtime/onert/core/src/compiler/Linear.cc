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

#include "backend/IShapeFixer.h"
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
  {
    const ir::Graph &graph = lowered_graph.graph();
    const ir::OpSequences &op_seqs = lowered_graph.op_seqs();
    const ir::Operands &operands = graph.operands();
    // op_seqs can't access a op_seq by an operand so that input_to_op_seqs can offer it
    std::unordered_map<ir::OperandIndex, std::list<ir::OpSequenceIndex>> input_to_op_seqs;

    // Get the relations between input/op_seq to be used for dfs-post-iter
    //
    //      [0]               # input -> _input_to_op_seqs[0] = {OP_SEQS0}
    //       |
    //     [OP_SEQS0]
    //       |
    //      [1]---------.     # input -> _input_to_op_seqs[1] = {OP_SEQS1, OP_SEQS2}
    //       |          |
    //  [OP_SEQS1]  [OP_SEQS2]
    //       |          |
    //      [2]        [3]    # input -> _input_to_op_seqs[2] = {OP_SEQS3}
    //       \         /      # input -> _input_to_op_seqs[3] = {OP_SEQS3}
    //       [OP_SEQS3]
    //            |
    //           [4]
    op_seqs.iterate([&](const ir::OpSequenceIndex &op_seq_idx, const ir::OpSequence &op_seq) {
      for (auto input : op_seq.getInputs())
      {
        // only valid_inputs
        const auto &operand = operands.at(input);
        if (operand.isConstant())
          continue;

        auto it = input_to_op_seqs.find(input);
        if (it == input_to_op_seqs.end())
        {
          std::list<ir::OpSequenceIndex> list{op_seq_idx};
          input_to_op_seqs[input] = list;
        }
        else
        {
          it->second.push_back(op_seq_idx);
        }
      }
    });

    std::unordered_map<ir::OpSequenceIndex, bool> visited;
    op_seqs.iterate(
        [&](const ir::OpSequenceIndex &index, const ir::OpSequence &) { visited[index] = false; });

    std::function<void(const ir::OpSequenceIndex &, const ir::OpSequence &)> dfs_recursive =
        [&](const ir::OpSequenceIndex &index, const ir::OpSequence &op_seq) -> void {
      if (visited[index])
        return;
      visited[index] = true;

      // The outputs should be not constants
      for (auto output : op_seq.getOutputs())
      {
        const auto it = input_to_op_seqs.find(output);
        if (it != input_to_op_seqs.end())
        {
          const auto &op_seq_index_list = it->second;
          for (const auto &index : op_seq_index_list)
          {
            auto &op_seq = op_seqs.at(index);
            dfs_recursive(index, op_seq);
          }
        }
      }

      order.emplace_back(index);
    };

    op_seqs.iterate(dfs_recursive);

    // All of the nodes must have been visited.
    assert(
        std::all_of(visited.begin(), visited.end(),
                    [](const std::pair<const ir::OpSequenceIndex, bool> &v) { return v.second; }));

    // NOTE. Now these op_seq are on the reverse order
    std::reverse(order.begin(), order.end());
  }
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
      VERBOSE(Linear) << "* OP_SEQ " << toString(lower_info->backend()) << " " << op_seq.getStr()
                      << std::endl;
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
    if ((lowered_graph.graph().getInputs() + lowered_graph.graph().getOutputs()).contains(ind))
      return;
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

    uses_map[ind] = obj.getUses().size();
    def_map[ind] = obj.getDef().size(); // should be 1 or 0

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
      tensor_builder->registerTensorInfo(ind, info, backend_layout, is_const);
    }

    tensor_builder_map[ind] = tensor_builder;
  });

  // If a tensor is model output, increase the use of the tensor.
  // This aim is same to above one.
  /*
  for (const auto &ind : graph.getOutputs())
  {
    uses_map[ind]++;
  }
  */

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
  /*
  VERBOSE(LINEAR) << "TENSORS as MODEL INPUT" << std::endl;
  for (const auto &ind : graph.getInputs())
  {
    auto tensor_builder = tensor_builder_map[ind];
    if (!tensor_builder) // for GeneratedTests.xxx_weights_as_inputs
      continue;
    tensor_builder->notifyFirstUse(ind);
  }
  */

  // At each operation,
  // 1. Scan DEF of outputs. If the DEF, allocate it
  // 2. Scan USE of inputs. Decrease the USE and deallocate if the USE is 0
  VERBOSE(LINEAR) << "TENSORS" << std::endl;
  for (const auto op_seq_ind : order)
  {
    const auto &op_seq = lowered_graph.op_seqs().at(op_seq_ind);
    for (const auto &op : op_seq.operations())
    {
      for (const auto &ind : op.node->getOutputs())
      {
        if (lowered_graph.graph().getOutputs().contains(ind))
          continue;
        assert(def_map.find(ind) != def_map.end());
        if (def_map[ind])
        {
          def_map[ind] = 0;
          tensor_builder_map[ind]->notifyFirstUse(ind);
        }
      }

      for (const auto &ind : op.node->getInputs())
      {
        if (lowered_graph.graph().getInputs().contains(ind))
          continue;
        assert(uses_map.find(ind) != uses_map.end());
        assert(uses_map[ind] > 0);
        uses_map[ind]--;
        if (uses_map[ind] == 0)
        {
          tensor_builder_map[ind]->notifyLastUse(ind);
        }
      }
    }
  }

  // Dispose and validate
  /*
  for (const auto &ind : graph.getOutputs())
  {
    --uses_map[ind];
    if (uses_map[ind] == 0) // To prevent notifyLastUse from being called twice
    {
      tensor_builder_map[ind]->notifyLastUse(ind);
    }
  }
  */

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
