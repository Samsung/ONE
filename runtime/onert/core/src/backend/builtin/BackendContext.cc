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
  return cpu_common::genTensors(*this, order, op_seqs, lower_info);
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
