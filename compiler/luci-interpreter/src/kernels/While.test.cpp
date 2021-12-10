/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "kernels/Less.h"
#include "kernels/While.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

RuntimeGraph *buildCondSubgraph(RuntimeModule *module, const std::vector<Tensor *> &input_prototypes, Tensor *input_cond,
                                IMemoryManager *memory_manager)
{
  RuntimeGraph *graph = module->addGraph(memory_manager);
  std::vector<Tensor *> input_tensors;
  for (const auto &proto: input_prototypes)
  {
    Tensor *input =
      graph->addTensor(std::make_unique<Tensor>(proto->element_type(), proto->shape(), AffineQuantization{}, ""));
    memory_manager->allocate_memory(*input);
    input_tensors.push_back(input);
  }
  Tensor *output =
    graph->addTensor(std::make_unique<Tensor>(DataType::BOOL, Shape{}, AffineQuantization{}, ""));

  memory_manager->allocate_memory(*output);

  graph->setInputTensors(input_tensors);
  graph->setOutputTensors({output});

  graph->addKernel(std::make_unique<Less>(input_tensors[0], input_cond, output));

  return graph;
}

RuntimeGraph *buildBodySubgraph(RuntimeModule *module, const std::vector<Tensor *> &prototypes, Tensor *input_add,
                                IMemoryManager *memory_manager)
{
  RuntimeGraph *graph = module->addGraph(memory_manager);
  std::vector<Tensor *> input_tensors;
  std::vector<Tensor *> output_tensors;
  for (const auto *proto: prototypes)
  {
    Tensor *input =
      graph->addTensor(std::make_unique<Tensor>(proto->element_type(), Shape{}, AffineQuantization{}, ""));
    Tensor *output =
      graph->addTensor(std::make_unique<Tensor>(proto->element_type(), Shape{}, AffineQuantization{}, ""));

    memory_manager->allocate_memory(*input);
    memory_manager->allocate_memory(*output);
    input_tensors.push_back(input);
    output_tensors.push_back(output);
  }

  graph->setInputTensors(input_tensors);
  graph->setOutputTensors(output_tensors);

  AddParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Add>(input_tensors[0], input_add, output_tensors[0], params));

  return graph;
}

TEST(WhileTest, FloatLoop10)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input = makeInputTensor<DataType::FLOAT32>({1}, {1}, memory_manager.get());
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  Tensor input_cond = makeInputTensor<DataType::FLOAT32>({1}, {10}, memory_manager.get());
  Tensor input_add = makeInputTensor<DataType::FLOAT32>({1}, {1}, memory_manager.get());

  RuntimeModule module(nullptr);
  RuntimeGraph *main_graph = module.addGraph(memory_manager.get());
  RuntimeGraph *cond_graph =
    buildCondSubgraph(&module, {&input}, &input_cond, memory_manager.get());
  RuntimeGraph *body_graph =
    buildBodySubgraph(&module, {&input}, &input_add, memory_manager.get());

  While kernel({&input}, {&output}, cond_graph, body_graph, main_graph);
  kernel.configure();
  memory_manager->allocate_memory(output);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({10}));
}

TEST(WhileTest, MultipleVariables)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor counter_begin = makeInputTensor<DataType::S32>({1}, {0}, memory_manager.get());
  Tensor sum = makeInputTensor<DataType::S32>({1}, {0}, memory_manager.get());
  Tensor counter_output = makeOutputTensor(DataType::S32);
  Tensor sum_output = makeOutputTensor(DataType::S32);

  Tensor counter_bound = makeInputTensor<DataType::S32>({1}, {10}, memory_manager.get());
  Tensor input_add = makeInputTensor<DataType::S32>({1}, {1}, memory_manager.get());

  RuntimeModule module(nullptr);
  RuntimeGraph *main_graph = module.addGraph(memory_manager.get());
  RuntimeGraph *cond_graph =
    buildCondSubgraph(&module, {&counter_begin, &sum}, &counter_bound, memory_manager.get());
  RuntimeGraph *body_graph =
    buildBodySubgraph(&module, {&counter_begin, &sum}, &input_add, memory_manager.get());

  AddParams params{};
  params.activation = Activation::NONE;
  body_graph->addKernel(std::make_unique<Add>(body_graph->getInputTensors()[0], body_graph->getInputTensors()[1], body_graph->getOutputTensors()[1], params));

  While kernel({&counter_begin, &sum}, {&counter_output, &sum_output}, cond_graph, body_graph, main_graph);
  kernel.configure();
  memory_manager->allocate_memory(counter_output);
  memory_manager->allocate_memory(sum_output);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int32_t>(sum_output)[0], 45);
}

TEST(WhileTest, LargeOutput)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor counter_begin = makeInputTensor<DataType::S32>({1}, {0}, memory_manager.get());
  Tensor large_tensor(DataType::S32, {1000, 1000}, {}, "");
  memory_manager->allocate_memory(large_tensor);

  Tensor counter_output = makeOutputTensor(DataType::S32);
  Tensor output = makeOutputTensor(DataType::S32);

  Tensor counter_bound = makeInputTensor<DataType::S32>({1}, {10}, memory_manager.get());
  Tensor input_add = makeInputTensor<DataType::S32>({1}, {1}, memory_manager.get());

  RuntimeModule module(nullptr);
  RuntimeGraph *main_graph = module.addGraph(memory_manager.get());
  RuntimeGraph *cond_graph =
    buildCondSubgraph(&module, {&counter_begin, &large_tensor}, &counter_bound, memory_manager.get());
  RuntimeGraph *body_graph =
    buildBodySubgraph(&module, {&counter_begin, &large_tensor}, &input_add, memory_manager.get());

  AddParams params{};
  params.activation = Activation::NONE;
  body_graph->addKernel(std::make_unique<Add>(body_graph->getInputTensors()[0], body_graph->getInputTensors()[1], body_graph->getOutputTensors()[1], params));

  While kernel({&counter_begin, &large_tensor}, {&counter_output, &output}, cond_graph, body_graph, main_graph);
  kernel.configure();
  memory_manager->allocate_memory(counter_output);
  memory_manager->allocate_memory(output);
  kernel.execute();

  SUCCEED();
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
