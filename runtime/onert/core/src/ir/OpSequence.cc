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
#include "ir/OperationVisitor.h"
#include <sstream>

namespace onert
{
namespace ir
{

OpSequence::OpSequence(Layout layout) : _layout{layout}
{
  // DO NOTHING
}

void OpSequence::accept(OperationVisitor &v) const { v.visit(*this); }

// TODO: Impl Dumper instead of this method
std::string OpSequence::getStr() const
{
  // "  OpSequence IN(0,1,2) -> { op0(0,1,2:3), op1(3:4), op2(4:5) } -> OUT(5)"
  std::stringstream ss;
  ss << "  OpSequence IN(";
  for (const auto &index : getInputs())
  {
    ss << " " << index.value();
  }
  ss << " ) -> {";
  for (const auto &elem : _operations)
  {
    // input string
    std::string inputs;
    for (const auto &ind : elem.node->getInputs())
    {
      inputs += std::to_string(ind.value());
      inputs.push_back(',');
    }
    if (inputs.back() == ',')
      inputs.pop_back();

    // output string
    std::string outputs;
    for (const auto &ind : elem.node->getOutputs())
    {
      outputs += std::to_string(ind.value());
      outputs.push_back(',');
    }
    if (outputs.back() == ',')
      outputs.pop_back();

    ss << " " << elem.index.value() << "(" << elem.node->name() << ":" << inputs << ":" << outputs
       << ")";
  }
  ss << " } -> OUT(";
  for (const auto &index : getOutputs())
  {
    ss << " " << index.value();
  }
  ss << " )";
  return ss.str();
}

void OpSequence::remove(const OperationIndex &index)
{
  assert(exist(index));
  for (auto it = _operations.cbegin(); it != _operations.cend(); ++it)
  {
    if (it->index == index)
    {
      _operations.erase(it);
      break;
    }
  }
}

bool OpSequence::exist(const OperationIndex &index) const
{
  for (const auto &element : _operations)
  {
    if (element.index == index)
    {
      return true;
    }
  }
  return false;
}

} // namespace ir
} // namespace onert
