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

#include "ir/Graph.h"
#include "ir/verifier/Verifier.h"
#include <memory>
#include "../MockNode.h"

#include <typeindex>

namespace
{

using IndexSet = onert::ir::OperandIndexSequence;
using Mock = onert_test::ir::SimpleMock;

} // namespace

TEST(ir_Operand, neg_usedef)
{
  onert::ir::Graph graph;
  onert::ir::verifier::DAGChecker verifier;

  onert::ir::Shape shape(3);
  onert::ir::TypeInfo type{onert::ir::DataType::INT32};

  // Model Input/Output
  auto input_operand = graph.addOperand(shape, type);
  auto output_operand = graph.addOperand(shape, type);

  graph.addInput(input_operand);
  graph.addOutput(output_operand);

  // MockNode1
  auto operand_index1 = graph.addOperand(shape, type);
  auto mocknode_index1 =
    graph.addOperation(std::make_unique<Mock>(IndexSet{input_operand}, IndexSet{operand_index1}));

  // MockNode2
  auto operand_index2 = graph.addOperand(shape, type);
  auto mocknode_index2 =
    graph.addOperation(std::make_unique<Mock>(IndexSet{input_operand}, IndexSet{operand_index2}));

  // MockNode3(two input)
  auto multiinput_index = graph.addOperation(
    std::make_unique<Mock>(IndexSet{operand_index1, operand_index2}, IndexSet{output_operand}));

  graph.finishBuilding();

  ASSERT_TRUE(verifier.verify(graph));

  // Check def
  ASSERT_EQ(graph.operands().at(operand_index1).getDef(), mocknode_index1);
  ASSERT_EQ(graph.operands().at(operand_index2).getDef(), mocknode_index2);
  ASSERT_EQ(graph.operands().at(output_operand).getDef(), multiinput_index);

  ASSERT_NE(graph.operands().at(operand_index1).getDef(), mocknode_index2);
  ASSERT_NE(graph.operands().at(operand_index1).getDef(), multiinput_index);

  // Check use
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index1), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(mocknode_index2), true);
  ASSERT_EQ(graph.operands().at(input_operand).getUses().contains(multiinput_index), false);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().contains(multiinput_index), true);
  ASSERT_EQ(graph.operands().at(operand_index2).getUses().contains(multiinput_index), true);

  ASSERT_EQ(graph.operands().at(input_operand).getUses().size(), 2);
  ASSERT_EQ(graph.operands().at(operand_index1).getUses().size(), 1);
  ASSERT_EQ(graph.operands().at(output_operand).getUses().size(), 0);
}
