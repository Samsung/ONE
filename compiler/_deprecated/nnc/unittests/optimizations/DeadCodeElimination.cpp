/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "passes/optimizations/DeadCodeElimination.h"
#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/OutputOp.h"

#include <gtest/gtest.h>

namespace
{
using namespace nnc;
using namespace mir;

TEST(DeadCodeEliminationTest, RemovesSingleNodes)
{
  Graph graph;
  graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}});
  graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}});

  DeadCodeElimination pass;
  pass.run(&graph);
  ASSERT_EQ(graph.getNodes().size(), 0);
}

TEST(DeadCodeEliminationTest, RemovesChainedNodes)
{
  Graph graph;
  auto c1 = graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}})->getOutput(0);
  auto c2 = graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}})->getOutput(0);
  auto sum = graph.create<ops::AddOp>(c1, c2)->getOutput(0);
  graph.create<ops::AddOp>(sum, sum);

  DeadCodeElimination pass;
  pass.run(&graph);
  ASSERT_EQ(graph.getNodes().size(), 0);
}

TEST(DeadCodeEliminationTest, PreservesInputNode)
{
  Graph graph;
  graph.create<ops::InputOp>(TensorType{DataType::FLOAT32, {}});

  DeadCodeElimination pass;
  pass.run(&graph);
  ASSERT_EQ(graph.getNodes().size(), 1);
}

TEST(DeadCodeEliminationTest, PreservesOutputNode)
{
  Graph graph;
  auto c = graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}})->getOutput(0);
  graph.create<ops::OutputOp>(c);

  DeadCodeElimination pass;
  pass.run(&graph);
  ASSERT_EQ(graph.getNodes().size(), 2);
}

TEST(DeadCodeEliminationTest, PreservesUsedNodes)
{
  Graph graph;
  auto c1 = graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}})->getOutput(0);
  auto c2 = graph.create<ops::ConstantOp>(TensorVariant{DataType::FLOAT32, {}})->getOutput(0);
  graph.create<ops::AddOp>(c1, c2);
  graph.create<ops::OutputOp>(c1);
  graph.create<ops::OutputOp>(c2);

  DeadCodeElimination pass;
  pass.run(&graph);
  ASSERT_EQ(graph.getNodes().size(), 4);
}

} // unnamed namespace
