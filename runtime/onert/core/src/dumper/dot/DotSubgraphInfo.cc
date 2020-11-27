/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DotSubgraphInfo.h"

#include <sstream>

namespace onert
{
namespace dumper
{
namespace dot
{

DotSubgraphInfo::DotSubgraphInfo(const ir::OpSequenceIndex &index, const ir::OpSequence &op_seq,
                                 const util::Set<ir::OperandIndex> &shown_operands,
                                 const ir::Operations &operations_ctx)
  : _index{index}
{
  for (const auto &op_idx : op_seq.operations())
  {
    _operations.insert(op_idx);
    const auto &node = operations_ctx.at(op_idx);
    for (auto o : node.getInputs())
    {
      // Must be a shown operand, not op_seq's inputs
      if (shown_operands.contains(o) && !op_seq.getInputs().contains(o))
      {
        _operands.insert(o);
      }
    }
    for (auto o : node.getOutputs())
    {
      // Must be a shown operand, not op_seq's inputs
      if (shown_operands.contains(o) && !op_seq.getOutputs().contains(o))
      {
        _operands.insert(o);
      }
    }
  }
}

} // namespace dot
} // namespace dumper
} // namespace onert
