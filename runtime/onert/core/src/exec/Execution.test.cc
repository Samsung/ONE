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

#include "exec/Execution.h"

#include "compiler/Compiler.h"
#include "compiler/CompilerFactory.h"
#include "ir/Graph.h"
#include "ir/operation/BinaryArithmetic.h"
#include "util/TracingCtx.h"

#include <gtest/gtest.h>
#include <thread>

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
    operation::BinaryArithmetic::Param param1;
    param1.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param1.activation = Activation::NONE;
    auto input_set1 = OperandIndexSequence{operand_lhs, operand_rhs1};
    auto output_set1 = OperandIndexSequence{operand_result1};
    graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set1, output_set1, param1));
    operation::BinaryArithmetic::Param param2;
    param2.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param2.activation = Activation::NONE;
    auto input_set2 = OperandIndexSequence{operand_result1, operand_rhs2};
    auto output_set2 = OperandIndexSequence{operand_result2};
    graph->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set2, output_set2, param2));
    // Identify model inputs and outputs
    graph->addInput(operand_lhs);
    graph->addInput(operand_rhs1);
    graph->addOutput(operand_result2);
    graph->verify();

    // Compile
    auto model = std::make_shared<onert::ir::Model>();
    model->push(onert::ir::SubgraphIndex{0}, graph);
    coptions = onert::compiler::CompilerOptions::fromGlobalConfig();
    onert::compiler::Compiler compiler{model, coptions.get()};
    artifact = compiler.compile();
  }

public:
  std::shared_ptr<Graph> graph;
  std::unique_ptr<onert::compiler::CompilerOptions> coptions;
  std::shared_ptr<onert::compiler::CompilerArtifact> artifact;
};

class CompiledMockUpMultiModel
{
public:
  CompiledMockUpMultiModel()
  {
    // Model0: a float elementwise add operation
    // Model0 input: lhs0, rhs0
    // Model0 output: add result (result0)

    // Model1: a qasymm8 elementwise add operation
    // Model1 input: result0, rhs1
    // Model1 output: add result (result1)

    // Model2: a float elementwise add operation
    // Model2 input: result0, result1
    // Model2 output: add result (result2)

    // constant: rhs2
    // result0 <= (lhs0 + rhs0)
    // result1 <= (result0 + rhs1)
    // result2 <= (result0 + result1)
    // lhs0, rhs0, rh1, result0, result1, result2 shape: {1, 2, 2, 1}
    // activation: none (constant)

    // Update edge information
    edges.pkg_inputs.emplace_back(ModelIndex{0}, SubgraphIndex{0}, IOIndex{0});
    edges.pkg_inputs.emplace_back(ModelIndex{0}, SubgraphIndex{0}, IOIndex{1});
    edges.pkg_outputs.emplace_back(ModelIndex{2}, SubgraphIndex{0}, IOIndex{0});
    // From
    const auto result0 = IODesc{ModelIndex{0}, SubgraphIndex{0}, IOIndex{0}};
    const auto result1 = IODesc{ModelIndex{1}, SubgraphIndex{0}, IOIndex{0}};
    // To
    const auto lhs1 = IODesc{ModelIndex{1}, SubgraphIndex{0}, IOIndex{0}};
    const auto lhs2 = IODesc{ModelIndex{2}, SubgraphIndex{0}, IOIndex{0}};
    const auto rhs2 = IODesc{ModelIndex{2}, SubgraphIndex{0}, IOIndex{1}};
    edges.edges.insert({result0, lhs1});
    edges.edges.insert({result0, lhs2});
    edges.edges.insert({result1, rhs2});

    for (size_t i = 0; i < 3; ++i)
    {
      graphs.emplace_back(std::make_shared<Graph>());
    }
    Shape shape{1, 2, 2, 1};

    // Model0's add operands (result1 <= lhs0 + rhs0)
    DataType types[3] = {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM, DataType::FLOAT32};
    auto operand_lhs0 = graphs[0]->addOperand(shape, TypeInfo{types[0]});
    auto operand_rhs0 = graphs[0]->addOperand(shape, TypeInfo{types[0]});
    auto operand_result0 = graphs[0]->addOperand(shape, TypeInfo{types[0]});

    // Model0's add operation
    operation::BinaryArithmetic::Param param0;
    param0.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param0.activation = Activation::NONE;
    auto input_set0 = OperandIndexSequence{operand_lhs0, operand_rhs0};
    auto output_set0 = OperandIndexSequence{operand_result0};
    graphs[0]->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set0, output_set0, param0));

    // Model0's inputs/outputs
    graphs[0]->addInput(operand_lhs0);
    graphs[0]->addInput(operand_rhs0);
    graphs[0]->addOutput(operand_result0);
    graphs[0]->verify();

    // Model1's add operands (result2 <= Model0 result + rhs1)
    // static float rhs1_data[4] = {3, 1, -1, 5};
    static uint8_t rhs1_data[4] = {131, 129, 127, 133};
    const float scale = 1;
    const int32_t zero_point = 128;
    auto operand_lhs1 = graphs[1]->addOperand(shape, TypeInfo{types[1], scale, zero_point});
    auto operand_rhs1 = graphs[1]->addOperand(shape, TypeInfo{types[1], scale, zero_point});
    auto operand_result1 = graphs[1]->addOperand(shape, TypeInfo{types[1], scale, zero_point});
    graphs[1]
      ->operands()
      .at(operand_rhs1)
      .data(std::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(&rhs1_data), 4));

    // Model1's add operation
    operation::BinaryArithmetic::Param param1;
    param1.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param1.activation = Activation::NONE;
    auto input_set1 = OperandIndexSequence{operand_lhs1, operand_rhs1};
    auto output_set1 = OperandIndexSequence{operand_result1};
    graphs[1]->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set1, output_set1, param1));

    // Model1's inputs/outputs
    graphs[1]->addInput(operand_lhs1);
    graphs[1]->addOutput(operand_result1);
    graphs[1]->verify();

    // Model2's additional operands (result3 <= Model0 result + Model1 result)
    auto operand_lhs2 = graphs[2]->addOperand(shape, TypeInfo{types[2]});
    auto operand_rhs2 = graphs[2]->addOperand(shape, TypeInfo{types[2]});
    auto operand_result2 = graphs[2]->addOperand(shape, TypeInfo{types[2]});

    // Model2's add operation
    operation::BinaryArithmetic::Param param2;
    param2.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param2.activation = Activation::NONE;
    auto input_set2 = OperandIndexSequence{operand_lhs2, operand_rhs2};
    auto output_set2 = OperandIndexSequence{operand_result2};
    graphs[2]->addOperation(
      std::make_unique<operation::BinaryArithmetic>(input_set2, output_set2, param2));

    // Model1's inputs/outputs
    graphs[2]->addInput(operand_lhs2);
    graphs[2]->addInput(operand_rhs2);
    graphs[2]->addOutput(operand_result2);
    graphs[2]->verify();

    // Compile
    compile();
  }

public:
  void compile()
  {
    auto nnpkg = std::make_shared<onert::ir::NNPkg>();
    coptions = onert::compiler::CompilerOptions::fromGlobalConfig();

    for (uint16_t i = 0; i < graphs.size(); ++i)
    {
      auto model = std::make_shared<onert::ir::Model>();
      model->push(SubgraphIndex{0}, graphs[i]);

      nnpkg->push(onert::ir::ModelIndex{i}, std::move(model));
    }
    for (const auto &pkg_input : edges.pkg_inputs)
    {
      nnpkg->addInput(pkg_input);
    }
    for (const auto &pkg_output : edges.pkg_outputs)
    {
      nnpkg->addOutput(pkg_output);
    }
    for (const auto &edge : edges.edges)
    {
      nnpkg->addEdge(edge.from, edge.to);
    }
    auto compiler = onert::compiler::CompilerFactory::get().create(nnpkg, coptions.get());
    nnpkg.reset();
    artifact = compiler->compile();
  }

public:
  std::vector<std::shared_ptr<Graph>> graphs;
  std::unique_ptr<onert::compiler::CompilerOptions> coptions;
  std::shared_ptr<onert::compiler::CompilerArtifact> artifact;
  ModelEdges edges;
};

TEST(ExecInstance, simple)
{
  auto mockup = CompiledMockUpModel();
  auto graph = mockup.graph;
  auto executors = mockup.artifact->_executors;

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
  execution.execute(onert::exec::ExecutionOptions{});

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

TEST(ExecInstance, twoCompile)
{
  auto mockup = CompiledMockUpModel();
  auto graph = mockup.graph;
  auto executors1 = mockup.artifact->_executors;
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
  auto model = std::make_shared<onert::ir::Model>();
  model->push(onert::ir::SubgraphIndex{0}, graph);
  auto coptions = onert::compiler::CompilerOptions::fromGlobalConfig();
  onert::compiler::Compiler compiler{model, coptions.get()};
  std::shared_ptr<onert::compiler::CompilerArtifact> artifact = compiler.compile();
  onert::exec::Execution execution2{artifact->_executors};

  const float exe2_input1_buffer[4] = {2, 1, -2, 0};
  const float exe2_input2_buffer[4] = {-3, 3, 1, 2};
  float exe2_output_buffer[4] = {};
  const float exe2_output_expected[4] = {2, 5, -2, 7};

  execution2.setInput(input1, reinterpret_cast<const void *>(exe2_input1_buffer), 16);
  execution2.setInput(input2, reinterpret_cast<const void *>(exe2_input2_buffer), 16);
  execution2.setOutput(output, reinterpret_cast<void *>(exe2_output_buffer), 16);

  onert::exec::ExecutionOptions exec_options;
  execution1.execute(exec_options);
  execution2.execute(exec_options);

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
  auto executors = mockup.artifact->_executors;
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

  onert::exec::ExecutionOptions exec_options;
  execution1.execute(exec_options);
  execution2.execute(exec_options);

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
            std::shared_ptr<onert::exec::IExecutors> &executors)
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

    execution.execute(onert::exec::ExecutionOptions{});
  }

private:
  const float (&_input1)[4];
  const float (&_input2)[4];
  float (&_output)[4];
  std::shared_ptr<onert::exec::IExecutors> &_executors;
};

// Support multi-thread execution
TEST(ExecInstance, twoThreads)
{
  auto mockup = CompiledMockUpModel();
  auto executors = mockup.artifact->_executors;

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
  auto executors = mockup.artifact->_executors;

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
  execution.startExecute(onert::exec::ExecutionOptions{});
  execution.waitFinish();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

TEST(ExecInstance, multi_model_simple)
{
  auto mockup = CompiledMockUpMultiModel();
  auto executors = mockup.artifact->_executors;

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float input1_buffer[4] = {1, 0, -1, -2};
  const float input2_buffer[4] = {1, -3, 2, -4};
  float output_buffer[4] = {};
  const float output_expected[4] = {7, -5, 1, -7};

  onert::exec::Execution execution{executors};

  execution.setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16);
  execution.setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16);
  execution.setOutput(output, reinterpret_cast<void *>(output_buffer), 16);
  execution.execute(onert::exec::ExecutionOptions{});

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

TEST(ExecInstance, multi_model_twoCompile)
{
  auto mockup = CompiledMockUpMultiModel();
  auto executors1 = mockup.artifact->_executors;
  onert::exec::Execution execution1{executors1};

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float exe1_input1_buffer[4] = {1, 0, -1, -2};
  const float exe1_input2_buffer[4] = {1, -3, 2, -4};
  float exe1_output_buffer[4] = {};
  const float exe1_output_expected[4] = {7, -5, 1, -7};

  execution1.setInput(input1, reinterpret_cast<const void *>(exe1_input1_buffer), 16);
  execution1.setInput(input2, reinterpret_cast<const void *>(exe1_input2_buffer), 16);
  execution1.setOutput(output, reinterpret_cast<void *>(exe1_output_buffer), 16);

  // Make new executor: compile again
  mockup.compile();
  onert::exec::Execution execution2{mockup.artifact->_executors};

  const float exe2_input1_buffer[4] = {2, 1, -2, 0};
  const float exe2_input2_buffer[4] = {-3, 3, 1, 2};
  float exe2_output_buffer[4] = {};
  const float exe2_output_expected[4] = {1, 9, -3, 9};

  execution2.setInput(input1, reinterpret_cast<const void *>(exe2_input1_buffer), 16);
  execution2.setInput(input2, reinterpret_cast<const void *>(exe2_input2_buffer), 16);
  execution2.setOutput(output, reinterpret_cast<void *>(exe2_output_buffer), 16);

  onert::exec::ExecutionOptions exec_options;
  execution1.execute(exec_options);
  execution2.execute(exec_options);

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(exe1_output_buffer[i], exe1_output_expected[i]);
    EXPECT_EQ(exe2_output_buffer[i], exe2_output_expected[i]);
  }
}

// Support two initialized execution instance then ordered execution
TEST(ExecInstance, multi_model_twoExecution)
{
  auto mockup = CompiledMockUpMultiModel();
  auto executors = mockup.artifact->_executors;
  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output1 = IOIndex{0};

  const float exe1_input1_buffer[4] = {1, 0, -1, -2};
  const float exe1_input2_buffer[4] = {1, -3, 2, -4};
  float exe1_output_buffer[4] = {};
  const float exe1_output_expected[4] = {7, -5, 1, -7};
  const float exe2_output_expected[4] = {1, 9, -3, 9};

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

  onert::exec::ExecutionOptions exec_options;
  execution1.execute(exec_options);
  execution1.execute(exec_options);
  execution2.execute(exec_options);
  execution2.execute(exec_options);

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(exe1_output_buffer[i], exe1_output_expected[i]);
    EXPECT_EQ(exe2_output_buffer[i], exe2_output_expected[i]);
  }
}

// Multi-model is not thread-safe yet

// Support asynchronous execution
TEST(ExecInstance, multi_model_async)
{
  auto mockup = CompiledMockUpMultiModel();
  auto executors = mockup.artifact->_executors;

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const float input1_buffer[4] = {1, 0, -1, -2};
  const float input2_buffer[4] = {1, -3, 2, -4};
  float output_buffer[4] = {};
  const float output_expected[4] = {7, -5, 1, -7};

  onert::exec::Execution execution{executors};

  execution.setInput(input1, reinterpret_cast<const void *>(input1_buffer), 16);
  execution.setInput(input2, reinterpret_cast<const void *>(input2_buffer), 16);
  execution.setOutput(output, reinterpret_cast<void *>(output_buffer), 16);
  execution.startExecute(onert::exec::ExecutionOptions{});
  execution.waitFinish();

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

TEST(ExecInstance, multi_model_dequant_input_quant_output)
{
  auto mockup = CompiledMockUpMultiModel();
  auto executors = mockup.artifact->_executors;

  auto input1 = IOIndex{0};
  auto input2 = IOIndex{1};
  auto output = IOIndex{0};

  const uint8_t input1_buffer[4] = {138, 128, 118, 108}; // {1, 0, -1, -2}
  const uint8_t input2_buffer[4] = {138, 98, 148, 88};   // {1, -3, 2, -4}
  uint8_t output_buffer[4] = {};
  const uint8_t output_expected[4] = {198, 78, 138, 58}; // {7, -5, 1, -7}
  float scale = 0.1;
  int32_t zero_point = 128;

  onert::exec::Execution execution{executors};

  onert::ir::TypeInfo type_info{onert::ir::DataType::QUANT_UINT8_ASYMM, scale, zero_point};
  execution.setInputType(input1, type_info);
  execution.setInput(input1, execution.getInputShape(input1),
                     reinterpret_cast<const void *>(input1_buffer), 4);
  execution.setInputType(input2, type_info);
  execution.setInput(input2, execution.getInputShape(input2),
                     reinterpret_cast<const void *>(input2_buffer), 4);
  execution.setOutputType(output, type_info);
  execution.setOutput(output, execution.getOutputShape(output),
                      reinterpret_cast<void *>(output_buffer), 4);
  execution.execute(onert::exec::ExecutionOptions{});

  for (auto i = 0; i < 4; i++)
  {
    EXPECT_EQ(output_buffer[i], output_expected[i]);
  }
}

// TODO Add an unittest multi_model_quant_input_dequant_output

} // namespace
