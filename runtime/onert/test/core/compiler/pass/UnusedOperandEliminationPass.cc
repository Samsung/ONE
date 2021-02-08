/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "compiler/pass/UnusedOperandEliminationPass.h"

using namespace onert::ir;
using namespace onert::compiler::pass;

TEST(UnusedOperandEliminationPass, Simple)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};
  auto in = graph.addOperand(shape, type);
  auto out = graph.addOperand(shape, type);

  auto unused = graph.addOperand(shape, type);

  // Set model inputs/outputs
  graph.addInput(in);
  graph.addOutput(out);

  UnusedOperandEliminationPass{graph}.run();

  ASSERT_TRUE(graph.operands().exist(in));
  ASSERT_TRUE(graph.operands().exist(out));
  ASSERT_FALSE(graph.operands().exist(unused));
}
