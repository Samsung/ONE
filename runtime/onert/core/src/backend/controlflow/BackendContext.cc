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

namespace onert
{
namespace backend
{
namespace controlflow
{

std::vector<std::pair<ir::OpSequenceIndex, std::unique_ptr<exec::FunctionSequence>>>
BackendContext::kernelGen(const std::vector<ir::OpSequenceIndex> &order,
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

} // namespace controlflow
} // namespace backend
} // namespace onert
