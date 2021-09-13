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

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

RuntimeGraph *buildAddSubgraph(RuntimeModule *module)
{
  RuntimeGraph *graph = module->addGraph(nullptr);
  Tensor *input1 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *input2 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *output = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));

  graph->setInputTensors({input1, input2});
  graph->setOutputTensors({output});

  AddParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Add>(input1, input2, output, params));

  return graph;
}

RuntimeGraph *buildMulSubgraph(RuntimeModule *module)
{
  RuntimeGraph *graph = module->addGraph(nullptr);
  Tensor *input1 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *input2 = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));
  Tensor *output = graph->addTensor(
    std::make_unique<Tensor>(DataType::FLOAT32, Shape{}, AffineQuantization{}, ""));

  graph->setInputTensors({input1, input2});
  graph->setOutputTensors({output});

  MulParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Mul>(input1, input2, output, params));

  return graph;
}

TEST(IfTest, CondTrue)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({1}, {true});
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7});
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2});
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module);
  RuntimeGraph *else_graph = buildMulSubgraph(&module);

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({6, 9}));
}

TEST(IfTest, CondFalse)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({1}, {false});
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7});
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2});
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module);
  RuntimeGraph *else_graph = buildMulSubgraph(&module);

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({5, 14}));
}

TEST(IfTest, InvalidCondType_NEG)
{
  Tensor cond = makeInputTensor<DataType::FLOAT32>({1}, {1});
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7});
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2});
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module);
  RuntimeGraph *else_graph = buildMulSubgraph(&module);

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(IfTest, InvalidCondElementNum_NEG)
{
  Tensor cond = makeInputTensor<DataType::BOOL>({2}, {false, true});
  Tensor input1 = makeInputTensor<DataType::FLOAT32>({2}, {5, 7});
  Tensor input2 = makeInputTensor<DataType::FLOAT32>({1, 2}, {1, 2});
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  RuntimeModule module(nullptr);
  RuntimeGraph *then_graph = buildAddSubgraph(&module);
  RuntimeGraph *else_graph = buildMulSubgraph(&module);

  If kernel(&cond, {&input1, &input2}, {&output}, then_graph, else_graph);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
