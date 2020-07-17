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

#include "ir/OpSequence.h"

#include "ir/Operations.h"
#include "ir/OperationVisitor.h"
#include <sstream>

namespace
{

std::string getStrFromIndice(const onert::ir::OperandIndexSequence &indice)
{
  std::string str;
  for (const auto &ind : indice)
  {
    str += std::to_string(ind.value());
    str.push_back(',');
  }
  if (str.back() == ',')
    str.pop_back();

  return str;
}
}

namespace onert
{
namespace ir
{

OpSequence::OpSequence(Layout layout) : _layout{layout}, _has_dynamic_tensor{false}
{
  // DO NOTHING
}

void OpSequence::accept(OperationVisitor &v) const { v.visit(*this); }

// TODO: Impl Dumper instead of this method
std::string getStrFromOpSeq(const OpSequence &op_seq, const Operations &operations)
{
  // "  OpSequence IN(0,1,2) -> { op0(0,1,2:3), op1(3:4), op2(4:5) } -> OUT(5)"
  std::stringstream ss;
  ss << "  OpSequence IN(" << getStrFromIndice(op_seq.getInputs()) << ") -> {";
  for (const auto &op_idx : op_seq)
  {
    ss << " " << op_idx.value() << "(" << operations.at(op_idx).name() << ":"
       << getStrFromIndice(operations.at(op_idx).getInputs()) << ":"
       << getStrFromIndice(operations.at(op_idx).getOutputs()) << ")";
  }
  ss << " } -> OUT(" << getStrFromIndice(op_seq.getOutputs()) << ")";
  return ss.str();
}

void OpSequence::remove(const OperationIndex &index)
{
  assert(exist(index));
  for (auto it = _operations.cbegin(); it != _operations.cend(); ++it)
  {
    if (*it == index)
    {
      _operations.erase(it);
      break;
    }
  }
}

bool OpSequence::exist(const OperationIndex &index) const
{
  for (const auto &inner_op_idx : _operations)
  {
    if (inner_op_idx == index)
    {
      return true;
    }
  }
  return false;
}

} // namespace ir
} // namespace onert
