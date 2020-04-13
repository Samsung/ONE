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

#include <gtest/gtest.h>

#include "ir/Operation.h"
#include "ir/Graph.h"
#include "ir/verifier/Verifier.h"
#include "memory"
#include "ir/Operand.h"
#include "../MockNode.h"

using IndexSet = neurun::ir::OperandIndexSequence;
using Mock = neurun_test::ir::SimpleMock;

TEST(Verifier, dag_checker)
{
  neurun::ir::Graph graph;

  neurun::ir::Shape shape{3};
  neurun::ir::TypeInfo type{neurun::ir::DataType::INT32};

  auto operand1 = graph.addOperand(shape, type);
  auto operand2 = graph.addOperand(shape, type);

  graph.addInput(operand1);
  graph.addOutput(operand2);

  graph.addOperation(std::make_unique<Mock>(IndexSet{operand1}, IndexSet{operand2}));

  graph.finishBuilding();

  neurun::ir::verifier::DAGChecker verifier;

  ASSERT_EQ(verifier.verify(graph), true);
}
