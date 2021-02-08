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

#include <gtest/gtest.h>

#include <memory>

#include "ir/Graph.h"
#include "interp/InterpExecutor.h"
#include "exec/Execution.h"
#include "ir/operation/BinaryArithmetic.h"

namespace
{

using namespace onert::ir;
using InterpExecutor = onert::interp::InterpExecutor;
using Execution = onert::exec::Execution;
using ExecutorMap = onert::exec::ExecutorMap;

class InterpExecutorTest : public ::testing::Test
{
protected:
  virtual void SetUp() {}
  void CreateSimpleModel()
  {
    // Model: one elementwise add operation
    // model input: lhs, rhs
    // model output: add result
    // lhs, rhs, result shape: {1, 2, 2, 1}
    // activation: none (constant)
    _graph = std::make_unique<Graph>();

    // Add operands

    Shape shape{1, 2, 2, 1};
    TypeInfo type{DataType::INT32};
    Shape shape_scalar(0);
    TypeInfo type_scalar{DataType::INT32};

    auto operand_lhs = _graph->addOperand(shape, type);
    auto operand_rhs = _graph->addOperand(shape, type);
    auto operand_result = _graph->addOperand(shape, type);

    // Add operations

    operation::BinaryArithmetic::Param param;
    param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param.activation = Activation::NONE;
    auto input_set = OperandIndexSequence{operand_lhs, operand_rhs};
    auto output_set = OperandIndexSequence{operand_result};
    _graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set, output_set, param));

    // Identify model inputs and outputs

    _graph->getInputs().append(operand_lhs);
    _graph->getInputs().append(operand_rhs);
    _graph->getOutputs().append(operand_result);

    _graph->verify();

    auto subgs = std::make_shared<onert::ir::Subgraphs>();
    subgs->push(onert::ir::SubgraphIndex{0}, _graph);
    _graph->setSubgraphs(subgs);

    _executors = std::make_shared<ExecutorMap>();
    _executors->insert(
      std::make_pair(onert::ir::SubgraphIndex{0}, std::make_unique<InterpExecutor>(*_graph)));
  }

  void CreateTwoStepModel()
  {
    // Model: two elementwise add operation
    // model input: lhs, rhs1
    // model output: second add result (result2)
    // constant: rhs2
    // result1 <= (lhs + rhs)
    // result2 <= (result1 + rhs2)
    // lhs, rhs1, rh2, result1, result2 shape: {1, 2, 2, 1}
    // activation: none (constant)
    _graph = std::make_unique<Graph>();

    // 1st add operands (result1 <= lhs + rhs1)

    Shape shape{1, 2, 2, 1};
    TypeInfo type{DataType::INT32};
    Shape shape_scalar(0);
    TypeInfo type_scalar{DataType::INT32};

    static int32_t rhs2_data[4] = {3, 1, -1, 5};

    auto operand_lhs = _graph->addOperand(shape, type);
    auto operand_rhs1 = _graph->addOperand(shape, type);
    auto operand_result1 = _graph->addOperand(shape, type);
    auto operand_rhs2 = _graph->addOperand(shape, type);
    auto operand_result2 = _graph->addOperand(shape, type);
    _graph->operands()
      .at(operand_rhs2)
      .data(std::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(&rhs2_data), 16));

    // 2nd add operations (result2 <= result1 + rhs2)

    operation::BinaryArithmetic::Param param1;
    param1.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param1.activation = Activation::NONE;
    auto input_set1 = OperandIndexSequence{operand_lhs, operand_rhs1};
    auto output_set1 = OperandIndexSequence{operand_result1};
    _graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set1, output_set1, param1));

    operation::BinaryArithmetic::Param param2;
    param2.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param2.activation = Activation::NONE;
    auto input_set2 = OperandIndexSequence{operand_result1, operand_rhs2};
    auto output_set2 = OperandIndexSequence{operand_result2};
    _graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set2, output_set2, param2));

    // Identify model inputs and outputs

    _graph->getInputs().append(operand_lhs);
    _graph->getInputs().append(operand_rhs1);
    _graph->getOutputs().append(operand_result2);

    _graph->verify();

    auto subgs = std::make_shared<onert::ir::Subgraphs>();
    subgs->push(onert::ir::SubgraphIndex{0}, _graph);
    _graph->setSubgraphs(subgs);

    _executors = std::make_shared<ExecutorMap>();
    _executors->insert(
      std::make_pair(onert::ir::SubgraphIndex{0}, std::make_unique<InterpExecutor>(*_graph)));
  }

  void CreateUnspecifiedDimensionsModel()
  {
    // Model: one elementwise add operation
    // model input: lhs, rhs
    // model output: add result
    // lhs, rhs, result shape: {1, unknown, 2, 1}
    // activation: none (constant)
    _graph = std::make_unique<Graph>();

    // Add operands

    Shape shape{1, 0, 2, 1};
    TypeInfo type{DataType::INT32};
    Shape shape_scalar(0);
    TypeInfo type_scalar{DataType::INT32};

    auto operand_lhs = _graph->addOperand(shape, type);
    auto operand_rhs = _graph->addOperand(shape, type);

    auto operand_activation = _graph->addOperand(shape_scalar, type_scalar);
    _graph->operands()
      .at(operand_activation)
      .data(std::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(&_activation_value), 4));

    auto operand_result = _graph->addOperand(shape, type);

    // Add operations

    operation::BinaryArithmetic::Param param;
    param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param.activation = Activation::NONE;
    auto input_set = OperandIndexSequence{operand_lhs, operand_rhs};
    auto output_set = OperandIndexSequence{operand_result};
    _graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set, output_set, param));

    // Identify model inputs and outputs

    _graph->getInputs().append(operand_lhs);
    _graph->getInputs().append(operand_rhs);
    _graph->getOutputs().append(operand_result);

    _graph->verify();

    auto subgs = std::make_shared<onert::ir::Subgraphs>();
    subgs->push(onert::ir::SubgraphIndex{0}, _graph);
    _graph->setSubgraphs(subgs);

    _executors = std::make_shared<ExecutorMap>();
    _executors->insert(
      std::make_pair(onert::ir::SubgraphIndex{0}, std::make_unique<InterpExecutor>(*_graph)));
  }

  void createExecution() { _execution = std::make_unique<Execution>(_executors); }

  virtual void TearDown() { _executors = nullptr; }

  std::shared_ptr<Graph> _graph{nullptr};
  std::shared_ptr<ExecutorMap> _executors{nullptr};
  std::unique_ptr<Execution> _execution{nullptr};
  const int32_t _activation_value{0};
};

TEST_F(InterpExecutorTest, create_empty)
{
  Graph graph;
  graph.verify();
  auto executor = std::make_unique<InterpExecutor>(graph);
  ASSERT_NE(executor, nullptr);
}

TEST_F(InterpExecutorTest, create_simple)
{
  CreateSimpleModel();
  ASSERT_NE(_executors, nullptr);
  ASSERT_NE(_executors->at(onert::ir::SubgraphIndex{0}), nullptr);
}

TEST_F(InterpExecutorTest, neg_setInput)
{
  CreateSimpleModel();
  createExecution();

  auto input1 = IOIndex{0};
  const int32_t input1_buffer[4] = {1, 0, -1, -2};

  EXPECT_THROW(_execution->setInput(input1, reinterpret_cast<const void *>(input1_buffer), 4),
               std::runtime_error);
  EXPECT_THROW(_execution->setInput(input1, reinterpret_cast<const void *>(input1_buffer), 12),
               std::runtime_error);
  EXPECT_NO_THROW(_execution->setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16));
}

TEST_F(InterpExecutorTest, neg_setOutput)
{
  CreateSimpleModel();
  createExecution();

  auto output = IOIndex{0};
  auto output_idx = _graph->getOutputs().at(output);

  int32_t output_buffer[4] = {};

  EXPECT_THROW(_execution->setOutput(output, reinterpret_cast<void *>(output_buffer), 4),
               std::runtime_error);
  EXPECT_THROW(_execution->setOutput(output, reinterpret_cast<void *>(output_buffer), 12),
               std::runtime_error);
  EXPECT_NO_THROW(_execution->setOutput(output, reinterpret_cast<void *>(output_buffer), 16));
}

TEST_F(InterpExecutorTest, neg_setInputForUnspecifiedDimensions)
{
  CreateUnspecifiedDimensionsModel();
  createExecution();

  auto input1 = IOIndex{0};
  const int32_t input1_buffer[4] = {1, 0, -1, -2};

  TypeInfo operand_type{DataType::INT32};
  Shape operand_shape{1, 2, 2, 1};

  EXPECT_THROW(_execution->setInput(input1, operand_type, operand_shape,
                                    reinterpret_cast<const void *>(input1_buffer), 4),
               std::runtime_error);
  EXPECT_THROW(_execution->setInput(input1, operand_type, operand_shape,
                                    reinterpret_cast<const void *>(input1_buffer), 12),
               std::runtime_error);
  EXPECT_NO_THROW(_execution->setInput(input1, operand_type, operand_shape,
                                       reinterpret_cast<const void *>(input1_buffer), 16));
}

TEST_F(InterpExecutorTest, neg_setOutputForUnspecifiedDimensions)
{
  CreateUnspecifiedDimensionsModel();
  createExecution();

  auto output = IOIndex{0};
  auto output_idx = _graph->getOutputs().at(output);

  TypeInfo operand_type{DataType::INT32};
  Shape operand_shape{1, 2, 2, 1};

  int32_t output_buffer[4] = {};

  EXPECT_THROW(_execution->setOutput(output, operand_type, operand_shape,
                                     reinterpret_cast<void *>(output_buffer), 4),
               std::runtime_error);
  EXPECT_THROW(_execution->setOutput(output, operand_type, operand_shape,
                                     reinterpret_cast<void *>(output_buffer), 12),
               std::runtime_error);
  EXPECT_NO_THROW(_execution->setOutput(output, operand_type, operand_shape,
                                        reinterpret_cast<void *>(output_buffer), 16));
}

TEST_F(InterpExecutorTest, execute)
{
  CreateSimpleModel();
  createExecution();

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto input1_idx = _graph->getInputs().at(input1);
  auto input2_idx = _graph->getInputs().at(input2);

  const int32_t input1_buffer[4] = {1, 0, -1, -2};
  const int32_t input2_buffer[4] = {1, -3, 2, -4};

  auto output = IOIndex{0};
  auto output_idx = _graph->getOutputs().at(output);

  int32_t output_buffer[4] = {};

  EXPECT_NO_THROW(_execution->setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16));
  EXPECT_NO_THROW(_execution->setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16));
  EXPECT_NO_THROW(_execution->setOutput(output, reinterpret_cast<void *>(output_buffer), 16));
  EXPECT_NO_THROW(_execution->execute());
  EXPECT_EQ(output_buffer[0], 2);
  EXPECT_EQ(output_buffer[1], -3);
  EXPECT_EQ(output_buffer[2], 1);
  EXPECT_EQ(output_buffer[3], -6);
}

TEST_F(InterpExecutorTest, executeTwoStep)
{
  CreateTwoStepModel();
  createExecution();

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto input1_idx = _graph->getInputs().at(input1);
  auto input2_idx = _graph->getInputs().at(input2);

  const int32_t input1_buffer[4] = {1, 0, -1, -2};
  const int32_t input2_buffer[4] = {1, -3, 2, -4};

  auto output = IOIndex{0};
  auto output_idx = _graph->getOutputs().at(output);

  int32_t output_buffer[4] = {};

  EXPECT_NO_THROW(_execution->setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16));
  EXPECT_NO_THROW(_execution->setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16));
  EXPECT_NO_THROW(_execution->setOutput(output, reinterpret_cast<void *>(output_buffer), 16));
  EXPECT_NO_THROW(_execution->execute());
  EXPECT_EQ(output_buffer[0], 5);
  EXPECT_EQ(output_buffer[1], -2);
  EXPECT_EQ(output_buffer[2], 0);
  EXPECT_EQ(output_buffer[3], -1);
}

} // namespace
