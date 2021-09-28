/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "core/RuntimeModule.h"
#include "kernels/Add.h"
#include "kernels/If.h"
#include "kernels/Mul.h"
#include "kernels/TestUtils.h"

#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class IfTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

RuntimeGraph *buildAddSubgraph(RuntimeModule *module, IMemoryManager *memory_manager)
{
  RuntimeGraph *graph = module->addGraph(memory_manager);
  Tensor *input1 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *input2 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *output = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));

  memory_manager->allocate_memory(*input1);
  memory_manager->allocate_memory(*input2);
  memory_manager->allocate_memory(*output);

  graph->setInputTensors({input1, input2});
  graph->setOutputTensors({output});

  AddParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Add>(input1, input2, output, params));

  return graph;
}

RuntimeGraph *buildMulSubgraph(RuntimeModule *module, IMemoryManager *memory_manager)
{
  RuntimeGraph *graph = module->addGraph(memory_manager);
  Tensor *input1 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *input2 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *output = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));

  memory_manager->allocate_memory(*input1);
  memory_manager->allocate_memory(*input2);
  memory_manager->allocate_memory(*output);

  graph->setInputTensors({input1, input2});
  graph->setOutputTensors({output});

  MulParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Mul>(input1, input2, output, params));

  return graph;
}

TEST_F(IfTest, CondTrue)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({1}, {true}, _memory_manager.get());
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7}, _memory_manager.get());
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2}, _memory_manager.get());
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module, _memory_manager.get());
  RuntimeGraph *else_graph = buildMulSubgraph(&module, _memory_manager.get());

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  kernel.configure();
  _memory_manager->allocate_memory(output);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({6, 9}));
}

TEST_F(IfTest, CondFalse)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({1}, {false}, _memory_manager.get());
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7}, _memory_manager.get());
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2}, _memory_manager.get());
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module, _memory_manager.get());
  RuntimeGraph *else_graph = buildMulSubgraph(&module, _memory_manager.get());

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  kernel.configure();
  _memory_manager->allocate_memory(output);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({5, 14}));
}

TEST_F(IfTest, InvalidCondType_NEG)
{
  Tensor cond = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7}, _memory_manager.get());
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2}, _memory_manager.get());
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module, _memory_manager.get());
  RuntimeGraph *else_graph = buildMulSubgraph(&module, _memory_manager.get());

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(IfTest, InvalidCondElementNum_NEG)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({2}, {false, true}, _memory_manager.get());
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7}, _memory_manager.get());
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2}, _memory_manager.get());
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module, _memory_manager.get());
  RuntimeGraph *else_graph = buildMulSubgraph(&module, _memory_manager.get());

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
