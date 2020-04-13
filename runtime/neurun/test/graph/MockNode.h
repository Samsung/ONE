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

#ifndef __NEURUN_TEST_GRAPH_MOCK_NODE_H__
#define __NEURUN_TEST_GRAPH_MOCK_NODE_H__

#include "ir/Operation.h"
#include "ir/OperandIndexSequence.h"

namespace neurun_test
{
namespace ir
{

class SimpleMock : public neurun::ir::Operation
{
public:
  SimpleMock(const neurun::ir::OperandIndexSequence &inputs,
             const neurun::ir::OperandIndexSequence &outputs)
      : Operation{neurun::ir::OperandConstraint::createAny()}
  {
    setInputs(inputs);
    setOutputs(outputs);
  }

public:
  void accept(neurun::ir::OperationVisitor &) const override {}
  neurun::ir::OpCode opcode() const final { return neurun::ir::OpCode::Invalid; }
};

} // namespace ir
} // namespace neurun_test

#endif // __NEURUN_TEST_GRAPH_MOCK_NODE_H__
