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
#include "ir/operation/BinaryArithmetic.h"
#include "ir/verifier/Verifier.h"

TEST(Graph, neg_inputs_and_outputs)
{
  onert::ir::Graph graph;

  onert::ir::OperandIndex index0{0u};
  onert::ir::OperandIndex index1{1u};

  graph.addInput({index0});
  graph.addInput({index1});

  onert::ir::OperandIndex index10{10u};
  onert::ir::OperandIndex index11{11u};
  onert::ir::OperandIndex index12{12u};

  graph.addOutput({index10});
  graph.addOutput({index11});
  graph.addOutput({index12});

  ASSERT_EQ(graph.getInputs().size(), 2);
  ASSERT_EQ(graph.getOutputs().size(), 3);

  onert::ir::IOIndex io_index0{0};
  onert::ir::IOIndex io_index1{1};
  onert::ir::IOIndex io_index2{2};

  ASSERT_EQ(graph.getInputs().at(io_index0), 0);
  ASSERT_EQ(graph.getInputs().at(io_index1), 1);

  ASSERT_EQ(graph.getOutputs().at(io_index0), 10);
  ASSERT_EQ(graph.getOutputs().at(io_index1), 11);
  ASSERT_EQ(graph.getOutputs().at(io_index2), 12);

  EXPECT_THROW(graph.getOutputs().at(onert::ir::IOIndex{3}), std::out_of_range);
}

using namespace onert::ir;

OperationIndex addAddOperation(Graph &graph, const OperandIndexSequence inputs,
                               const OperandIndexSequence outputs)
{
  // Add "ADD" operation
  operation::BinaryArithmetic::Param param;
  param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
  param.activation = Activation::NONE;
  return graph.addOperation(std::make_unique<operation::BinaryArithmetic>(inputs, outputs, param));
}

TEST(Graph, OneOpGraphSimpleValid)
{
  // Simple Graph with just one Add operation

  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto lhs = graph.addOperand(shape, type);
  auto rhs = graph.addOperand(shape, type);
  auto res = graph.addOperand(shape, type);

  addAddOperation(graph, {lhs, rhs}, {res});

  // Set model inputs/outputs
  graph.addInput(lhs);
  graph.addInput(rhs);
  graph.addOutput(res);

  graph.verify();

  SUCCEED();
}

TEST(Graph, neg_InvalidGraph_BadInput)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = graph.addOperand(shape, type);
  auto out = graph.addOperand(shape, type);

  // Set model inputs/outputs
  graph.addInput(in);
  graph.addOutput(out);
  graph.addInput(OperandIndex{89}); // Non-exisiting operand!

  EXPECT_ANY_THROW(graph.verify());
}

TEST(Graph, neg_InvalidGraph_BadOutput)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = graph.addOperand(shape, type);
  auto out = graph.addOperand(shape, type);

  // Set model inputs/outputs
  graph.addInput(in);
  graph.addOutput(out);
  graph.addOutput(OperandIndex{12}); // Non-exisiting operand!

  EXPECT_ANY_THROW(graph.verify());
}

TEST(Graph, neg_InvalidAddOperation_BadInputIndex)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto lhs = graph.addOperand(shape, type);
  auto rhs = graph.addOperand(shape, type);
  auto res = graph.addOperand(shape, type);

  // Set model inputs/outputs
  graph.addInput(lhs);
  graph.addInput(rhs);
  graph.addOutput(res);

  ASSERT_FALSE(addAddOperation(graph, {lhs, OperandIndex{99}}, {res}).valid());
}
