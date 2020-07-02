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
#include <memory>

#include <cassert>
#include <string>

namespace onert
{
namespace ir
{

OpSequenceIndex OpSequences::emplace(const OperationIndex &index, Layout layout)
{
  std::unique_ptr<OpSequence> op_seq = std::make_unique<OpSequence>(layout);
  op_seq->appendOperation(index);
  return push(std::move(op_seq));
}

OpSequenceIndex OpSequences::emplace(std::unique_ptr<OpSequence> &&op_seq)
{
  return push(std::move(op_seq));
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
void OpSequences::dump(const std::string &msg, const Operations &operations) const
{
  VERBOSE(OpSequences) << "OpSequences(" << msg << ")" << std::endl;
  iterate([&](const OpSequenceIndex &idx, const OpSequence &op_seq) {
    VERBOSE(OpSequences) << idx.value() << "] " << getStrFromOpSeq(op_seq, operations) << std::endl;
  });
}

void OpSequences::removeFromOpSequence(const OperationIndex &operation_index)
{
  const auto op_seq_index = findOperation(operation_index);
  auto &op_seq = at(op_seq_index);
  op_seq.remove(operation_index);
  if (op_seq.size() == 0)
  {
    remove(op_seq_index);
  }
}

OpSequenceIndex OpSequences::findOperation(const OperationIndex &operation_index) const
{
  for (auto &e : _objects)
  {
    OpSequence &object = *e.second;
    auto it = find(object.operations().begin(), object.operations().end(), operation_index);
    if (it != object.operations().end())
    {
      return e.first;
    }
  }
  throw std::runtime_error("Operation not found");
}

} // namespace ir
} // namespace onert
