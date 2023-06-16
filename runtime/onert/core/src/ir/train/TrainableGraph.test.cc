/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/train/TrainableGraph.h"
#include "ir/train/operation/Permute.h"

#include <gtest/gtest.h>

TEST(TrainableGraph, neg_inputs_and_outputs)
{
  onert::ir::train::TrainableGraph tgraph;

  onert::ir::OperandIndex index0{0u};
  onert::ir::OperandIndex index1{1u};

  tgraph.addInput({index0});
  tgraph.addInput({index1});

  onert::ir::OperandIndex index10{10u};

  tgraph.addOutput({index10});

  ASSERT_EQ(tgraph.getInputs().size(), 2);
  ASSERT_EQ(tgraph.getOutputs().size(), 1);

  onert::ir::IOIndex io_index0{0};
  onert::ir::IOIndex io_index1{1};

  ASSERT_EQ(tgraph.getInputs().at(io_index0), 0);
  ASSERT_EQ(tgraph.getInputs().at(io_index1), 1);

  ASSERT_EQ(tgraph.getOutputs().at(io_index0), 10);

  EXPECT_THROW(tgraph.getInputs().at(onert::ir::IOIndex{2}), std::out_of_range);
}

using namespace onert::ir;

OperationIndex addPermuteOperation(train::TrainableGraph &tgraph, const OperandIndex input,
                                   const OperandIndex output)
{
  // Add "Permute" operation
  auto op = std::make_unique<train::operation::Permute>(
    operation::Permute{input, output, operation::Permute::Type::COPY});
  return tgraph.addOperation(std::move(op));
}

TEST(TrainableGraph, OneOpGraphSimpleValid)
{
  // Simple Graph with just one Add operation

  train::TrainableGraph tgraph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = tgraph.addOperand(shape, type);
  auto out = tgraph.addOperand(shape, type);

  addPermuteOperation(tgraph, in, out);

  // Set model inputs/outputs
  tgraph.addInput(in);
  tgraph.addOutput(out);

  tgraph.verify();

  SUCCEED();
}

TEST(TrainableGraph, neg_InvalidGraph_BadInput)
{
  train::TrainableGraph tgraph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = tgraph.addOperand(shape, type);
  auto out = tgraph.addOperand(shape, type);

  // Set model inputs/outputs
  tgraph.addInput(in);
  tgraph.addOutput(out);
  tgraph.addInput(OperandIndex{89}); // Non-exisiting operand!

  EXPECT_ANY_THROW(tgraph.verify());
}

TEST(TrainableGraph, neg_InvalidGraph_BadOutput)
{
  train::TrainableGraph tgraph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = tgraph.addOperand(shape, type);
  auto out = tgraph.addOperand(shape, type);

  // Set model inputs/outputs
  tgraph.addInput(in);
  tgraph.addOutput(out);
  tgraph.addOutput(OperandIndex{12}); // Non-exisiting operand!

  EXPECT_ANY_THROW(tgraph.verify());
}

TEST(TrainableGraph, neg_InvalidOperation_BadInputIndex)
{
  train::TrainableGraph tgraph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = tgraph.addOperand(shape, type);
  auto out = tgraph.addOperand(shape, type);

  // Set model inputs/outputs
  tgraph.addInput(in);
  tgraph.addOutput(out);

  ASSERT_FALSE(addPermuteOperation(tgraph, OperandIndex{99}, out).valid());
}

OperationIndex replaceOperation(train::TrainableGraph &tgraph, const OperationIndex &index,
                                std::unique_ptr<train::ITrainableOperation> &&new_op)
{
  // Replace operation
  return tgraph.replaceOperation(index, std::move(new_op));
}

TEST(TrainableGraph, neg_InvalidOperation_BadReplacement)
{
  train::TrainableGraph tgraph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = tgraph.addOperand(shape, type);
  auto out = tgraph.addOperand(shape, type);

  const auto op_index = addPermuteOperation(tgraph, in, out);

  // Set model inputs/outputs
  tgraph.addInput(in);
  tgraph.addOutput(out);

  auto new_op = std::make_unique<train::operation::Permute>(
    operation::Permute{OperandIndex{99}, out, operation::Permute::Type::COPY});

  ASSERT_FALSE(replaceOperation(tgraph, op_index, std::move(new_op)).valid());
}
