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
#include "kernels/Minimum.h"
#include "kernels/While.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

RuntimeGraph *buildCondSubgraph(RuntimeModule *module, DataType dtype, Tensor *input_cond)
{
  RuntimeGraph *graph = module->addGraph();
  Tensor *input =
    graph->addTensor(std::make_unique<Tensor>(dtype, Shape{}, AffineQuantization{}, ""));
  Tensor *output =
    graph->addTensor(std::make_unique<Tensor>(DataType::BOOL, Shape{}, AffineQuantization{}, ""));

  graph->setInputTensors({input});
  graph->setOutputTensors({output});

  graph->addKernel(std::make_unique<Less>(input, input_cond, output));

  return graph;
}

RuntimeGraph *buildBodySubgraph(RuntimeModule *module, DataType dtype, Tensor *input_add)
{
  RuntimeGraph *graph = module->addGraph();
  Tensor *input =
    graph->addTensor(std::make_unique<Tensor>(dtype, Shape{}, AffineQuantization{}, ""));
  Tensor *output =
    graph->addTensor(std::make_unique<Tensor>(dtype, Shape{}, AffineQuantization{}, ""));

  graph->setInputTensors({input});
  graph->setOutputTensors({output});

  AddParams params{};
  params.activation = Activation::NONE;
  graph->addKernel(std::make_unique<Add>(input, input_add, output, params));

  return graph;
}

TEST(WhileTest, FloatLoop10)
{
  Tensor input = makeInputTensor<DataType::FLOAT32>({1}, {1});
  Tensor output = makeOutputTensor(DataType::FLOAT32);

  Tensor input_cond = makeInputTensor<DataType::FLOAT32>({1}, {10});
  Tensor input_add = makeInputTensor<DataType::FLOAT32>({1}, {1});

  RuntimeModule module(nullptr);
  RuntimeGraph *cond_graph = buildCondSubgraph(&module, DataType::FLOAT32, &input_cond);
  RuntimeGraph *body_graph = buildBodySubgraph(&module, DataType::FLOAT32, &input_add);

  While kernel({&input}, {&output}, cond_graph, body_graph);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear({10}));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
