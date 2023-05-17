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

#ifndef __ONERT_TEST_GRAPH_MOCK_NODE_H__
#define __ONERT_TEST_GRAPH_MOCK_NODE_H__

#include "ir/Operation.h"
#include "ir/OperandIndexSequence.h"

namespace onert_test
{
namespace ir
{

class SimpleMock : public onert::ir::Operation
{
public:
  SimpleMock(const onert::ir::OperandIndexSequence &inputs,
             const onert::ir::OperandIndexSequence &outputs)
    : Operation{onert::ir::OperandConstraint::createAny()}
  {
    setInputs(inputs);
    setOutputs(outputs);
  }

public:
  void accept(onert::ir::OperationVisitor &) const override {}
  void accept(onert::ir::MutableOperationVisitor &) override{};
  onert::ir::OpCode opcode() const final { return onert::ir::OpCode::Invalid; }
};

} // namespace ir
} // namespace onert_test

#endif // __ONERT_TEST_GRAPH_MOCK_NODE_H__
