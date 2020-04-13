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

#include "ir/OpSequences.h"
#include "util/logging.h"
#include "memory"

#include <cassert>
#include <string>

namespace neurun
{
namespace ir
{

OpSequenceIndex OpSequences::emplace(const OperationIndex &index, const Operation &node,
                                     Layout layout)
{
  std::unique_ptr<OpSequence> subg = std::make_unique<OpSequence>(layout);
  subg->appendOperation(index, node);
  return push(std::move(subg));
}

OpSequenceIndex OpSequences::emplace(std::unique_ptr<OpSequence> &&subg)
{
  return push(std::move(subg));
}

bool OpSequences::containsOperation(const OperationIndex &operation_index) const
{
  return findOperation(operation_index).valid();
}

OpSequenceIndex OpSequences::getOperation(const OperationIndex &operation_index) const
{
  OpSequenceIndex ret = findOperation(operation_index);
  assert(ret.valid());
  return ret;
}

// TODO: Extract this into external helper function
void OpSequences::dump(const std::string &msg) const
{
  VERBOSE(OpSequences) << "OpSequences(" << msg << ")" << std::endl;
  iterate([&](const OpSequenceIndex &idx, const OpSequence &subg) {
    VERBOSE(OpSequences) << idx.value() << "] " << subg.getStr() << std::endl;
  });
}

void OpSequences::removeFromSubgraph(const OperationIndex &operation_index)
{
  const auto op_seq_index = findOperation(operation_index);
  auto &subg = at(op_seq_index);
  subg.remove(operation_index);
  if (subg.size() == 0)
  {
    remove(op_seq_index);
  }
}

OpSequenceIndex OpSequences::findOperation(const OperationIndex &operation_index) const
{
  OpSequenceIndex ret;
  iterate([&](const OpSequenceIndex &index, const OpSequence &object) {
    for (const auto &elem : object.operations())
    {
      if (elem.index == operation_index)
        ret = index;
    }
  });
  return ret;
}

} // namespace ir
} // namespace neurun
