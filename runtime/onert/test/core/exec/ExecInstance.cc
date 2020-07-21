/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <thread>

#include "ir/Graph.h"
#include "compiler/Compiler.h"
#include "exec/Execution.h"
#include "ir/operation/Add.h"

namespace
{

using namespace onert::ir;

class CompiledMockUpModel
{
public:
  CompiledMockUpModel()
  {
    // Model: two elementwise add operation
    // model input: lhs, rhs1
    // model output: second add result (result2)
    // constant: rhs2
    // result1 <= (lhs + rhs)
    // result2 <= (result1 + rhs2)
    // lhs, rhs1, rh2, result1, result2 shape: {1, 2, 2, 1}
    // activation: none (constant)
    graph = std::make_shared<Graph>();
    // 1st add operands (result1 <= lhs + rhs1)
    Shape shape{1, 2, 2, 1};
    TypeInfo type{DataType::FLOAT32};
    static float rhs2_data[4] = {3, 1, -1, 5};
    auto operand_lhs = graph->addOperand(shape, type);
    auto operand_rhs1 = graph->addOperand(shape, type);
    auto operand_result1 = graph->addOperand(shape, type);
    auto operand_rhs2 = graph->addOperand(shape, type);
    auto operand_result2 = graph->addOperand(shape, type);
    graph->operands()
        .at(operand_rhs2)
        .data(std::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(&rhs2_data), 16));
    // 2nd add operations (result2 <= result1 + rhs2)
    operation::Add::Param param1;
    param1.activation = Activation::NONE;
    auto input_set1 = OperandIndexSequence{operand_lhs, operand_rhs1};
    auto output_set1 = OperandIndexSequence{operand_result1};
    graph->addOperation(std::make_unique<operation::Add>(input_set1, output_set1, param1));
    operation::Add::Param param2;
    param2.activation = Activation::NONE;
    auto input_set2 = OperandIndexSequence{operand_result1, operand_rhs2};
    auto output_set2 = OperandIndexSequence{operand_result2};
    graph->addOperation(std::make_unique<operation::Add>(input_set2, output_set2, param2));
    // Identify model inputs and outputs
    graph->addInput(operand_lhs);
    graph->addInput(operand_rhs1);
    graph->addOutput(operand_result2);
    graph->finishBuilding();

    // Compile
    auto subgs = std::make_shared<onert::ir::Subgraphs>();
    subgs->push(onert::ir::SubgraphIndex{0}, graph);
    onert::compiler::Compiler compiler{subgs};
    executors = compiler.compile();
  }

public:
  std::shared_ptr<Graph> graph;
  std::shared_ptr<onert::exec::ExecutorMap> executors;
};

TEST(ExecInstance, simple)
{
  auto mockup = CompiledMockUpModel();
  auto graph = mockup.graph;
  auto executors = mockup.executors;

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float input1_buffer[4] = {1, 0, -1, -2};
  const float input2_buffer[4] = {1, -3, 2, -4};
  float output_buffer[4] = {};
  const float output_expected[4] = {5, -2, 0, -1};

  onert::exec::Execution execution{executors};

  execution.setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16);
  execution.setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16);
  execution.setOutput(output, reinterpret_cast<void *>(output_buffer), 16);
  execution.execute();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

TEST(ExecInstance, twoCompile)
{
  auto mockup = CompiledMockUpModel();
  auto graph = mockup.graph;
  auto executors1 = mockup.executors;
  onert::exec::Execution execution1{executors1};

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float exe1_input1_buffer[4] = {1, 0, -1, -2};
  const float exe1_input2_buffer[4] = {1, -3, 2, -4};
  float exe1_output_buffer[4] = {};
  const float exe1_output_expected[4] = {5, -2, 0, -1};

  execution1.setInput(input1, reinterpret_cast<const void *>(exe1_input1_buffer), 16);
  execution1.setInput(input2, reinterpret_cast<const void *>(exe1_input2_buffer), 16);
  execution1.setOutput(output, reinterpret_cast<void *>(exe1_output_buffer), 16);

  // Make new executor: compile again
  auto subgs = std::make_shared<onert::ir::Subgraphs>();
  subgs->push(onert::ir::SubgraphIndex{0}, graph);
  onert::compiler::Compiler compiler{subgs};
  std::shared_ptr<onert::exec::ExecutorMap> executors2 = compiler.compile();
  onert::exec::Execution execution2{executors2};

  const float exe2_input1_buffer[4] = {2, 1, -2, 0};
  const float exe2_input2_buffer[4] = {-3, 3, 1, 2};
  float exe2_output_buffer[4] = {};
  const float exe2_output_expected[4] = {2, 5, -2, 7};

  execution2.setInput(input1, reinterpret_cast<const void *>(exe2_input1_buffer), 16);
  execution2.setInput(input2, reinterpret_cast<const void *>(exe2_input2_buffer), 16);
  execution2.setOutput(output, reinterpret_cast<void *>(exe2_output_buffer), 16);

  execution1.execute();
  execution2.execute();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(exe1_output_buffer[i], exe1_output_expected[i]);
    EXPECT_EQ(exe2_output_buffer[i], exe2_output_expected[i]);
  }
}

// Support two initialized execution instance then ordered execution
TEST(ExecInstance, twoExecution)
{
  auto mockup = CompiledMockUpModel();
  auto executors = mockup.executors;
  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output1 = IOIndex{0};

  const float exe1_input1_buffer[4] = {1, 0, -1, -2};
  const float exe1_input2_buffer[4] = {1, -3, 2, -4};
  float exe1_output_buffer[4] = {};
  const float exe1_output_expected[4] = {5, -2, 0, -1};
  const float exe2_output_expected[4] = {2, 5, -2, 7};

  onert::exec::Execution execution1{executors};
  execution1.setInput(input1, reinterpret_cast<const void *>(exe1_input1_buffer), 16);
  execution1.setInput(input2, reinterpret_cast<const void *>(exe1_input2_buffer), 16);
  execution1.setOutput(output1, reinterpret_cast<void *>(exe1_output_buffer), 16);

  const float exe2_input1_buffer[4] = {2, 1, -2, 0};
  const float exe2_input2_buffer[4] = {-3, 3, 1, 2};
  float exe2_output_buffer[4] = {};

  // Make new execution
  onert::exec::Execution execution2{executors};
  execution2.setInput(input1, reinterpret_cast<const void *>(exe2_input1_buffer), 16);
  execution2.setInput(input2, reinterpret_cast<const void *>(exe2_input2_buffer), 16);
  execution2.setOutput(output1, reinterpret_cast<void *>(exe2_output_buffer), 16);

  execution1.execute();
  execution2.execute();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(exe1_output_buffer[i], exe1_output_expected[i]);
    EXPECT_EQ(exe2_output_buffer[i], exe2_output_expected[i]);
  }
}

class Inference
{
public:
  Inference(const float (&input1)[4], const float (&input2)[4], float (&output)[4],
            std::shared_ptr<onert::exec::ExecutorMap> &executors)
      : _input1{input1}, _input2{input2}, _output{output}, _executors{executors}
  {
    // DO NOTHING
  }

  void inference(void)
  {
    auto input1 = IOIndex{0};
    auto input2 = IOIndex{1};
    auto output1 = IOIndex{0};

    onert::exec::Execution execution{_executors};
    execution.setInput(input1, reinterpret_cast<const void *>(_input1), 16);
    execution.setInput(input2, reinterpret_cast<const void *>(_input2), 16);
    execution.setOutput(output1, reinterpret_cast<void *>(_output), 16);

    execution.execute();
  }

private:
  const float (&_input1)[4];
  const float (&_input2)[4];
  float (&_output)[4];
  std::shared_ptr<onert::exec::ExecutorMap> &_executors;
};

// Support multi-thread execution
TEST(ExecInstance, twoThreads)
{
  auto mockup = CompiledMockUpModel();
  auto executors = mockup.executors;

  const float exe1_input1_buffer[4] = {1, 0, -1, -2};
  const float exe1_input2_buffer[4] = {1, -3, 2, -4};
  float exe1_output_buffer[4] = {};
  const float exe1_output_expected[4] = {5, -2, 0, -1};

  Inference execution1{exe1_input1_buffer, exe1_input2_buffer, exe1_output_buffer, executors};

  const float exe2_input1_buffer[4] = {2, 1, -2, 0};
  const float exe2_input2_buffer[4] = {-3, 3, 1, 2};
  float exe2_output_buffer[4] = {};
  const float exe2_output_expected[4] = {2, 5, -2, 7};

  Inference execution2{exe2_input1_buffer, exe2_input2_buffer, exe2_output_buffer, executors};

  std::thread t1{&Inference::inference, &execution1};
  std::thread t2{&Inference::inference, &execution2};

  t1.join();
  t2.join();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(exe1_output_buffer[i], exe1_output_expected[i]);
    EXPECT_EQ(exe2_output_buffer[i], exe2_output_expected[i]);
  }
}

// Support asynchronous execution
TEST(ExecInstance, async)
{
  auto mockup = CompiledMockUpModel();
  auto graph = mockup.graph;
  auto executors = mockup.executors;

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float input1_buffer[4] = {1, 0, -1, -2};
  const float input2_buffer[4] = {1, -3, 2, -4};
  float output_buffer[4] = {};
  const float output_expected[4] = {5, -2, 0, -1};

  onert::exec::Execution execution{executors};

  execution.setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16);
  execution.setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16);
  execution.setOutput(output, reinterpret_cast<void *>(output_buffer), 16);
  execution.startExecute();
  execution.waitFinish();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

} // namespace
