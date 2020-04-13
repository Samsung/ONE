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

#include "passes/optimizations/CombineTransposes.h"
#include "mir/ops/TransposeOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/OutputOp.h"
#include "Util.h"
#include <gtest/gtest.h>

using namespace std;
using namespace nnc;
using namespace mir;

namespace
{

TEST(OptPass, eliminateTransposesLinear)
{
  mir::Graph g;
  /*   Create graph:
   *      [input]
   *        ||
   *   [Transpose 1]
   *        ||
   *   [Transpose 2]
   *        ||
   *      [relu]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *tr1 = g.create<ops::TransposeOp>(input->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *tr15 = g.create<ops::TransposeOp>(tr1->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *tr2 = g.create<ops::TransposeOp>(tr15->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *relu = g.create<ops::ReluOp>(tr2->getOutput(0));

  // Check that layout is desired
  std::stringstream ss;
  DumpVisitor d(ss);
  CombineTransposes pass;
  pass.run(&g);
  g.accept(&d);
  // Assert only 1 transpose remains
  ASSERT_EQ("i_0.t_1.r_4.", ss.str());
}

TEST(OptPass, combineTransposesLinear)
{
  mir::Graph g;
  /* Create graph:
   *      [input]
   *        ||
   *   [Transpose 1]
   *        ||
   *   [Transpose 2]
   *        ||
   *      [relu]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *tr1 = g.create<ops::TransposeOp>(input->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *tr2 = g.create<ops::TransposeOp>(tr1->getOutput(0), vector<size_t>{0, 2, 1});
  Operation *relu = g.create<ops::ReluOp>(tr2->getOutput(0));

  std::stringstream ss;
  DumpVisitor d(ss);
  CombineTransposes pass;
  pass.run(&g);
  g.accept(&d);

  // Assert transposes are combined
  ASSERT_EQ("i_0.t_4.r_3.", ss.str());
  Operation::Use use = g.getInputs()[0]->getOutput(0)->getUses()[0];
  auto ax_ord_actual = dynamic_cast<ops::TransposeOp *>(use.getNode())->getAxisOrder();
  auto ax_ord_true = vector<size_t>{1, 2, 0};
  ASSERT_TRUE(ax_ord_actual == ax_ord_true);
}

TEST(OptPass, combineTransposesBush)
{
  mir::Graph g;
  /*      Create graph:
   *         [input]
   *            ||
   *       [Transpose 1]
   *        //       \\
   *[Transpose 2] [Transpose 3]
   *       \\       //
   *          [Add]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3, 2}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *tr1 = g.create<ops::TransposeOp>(input->getOutput(0), vector<size_t>{1, 0, 2, 3});
  Operation *tr2 = g.create<ops::TransposeOp>(tr1->getOutput(0), vector<size_t>{1, 0, 2, 3});
  Operation *tr3 = g.create<ops::TransposeOp>(tr1->getOutput(0), vector<size_t>{1, 0, 2, 3});
  Operation *elw = g.create<ops::AddOp>(tr2->getOutput(0), tr3->getOutput(0));
  std::stringstream ss;
  DumpVisitor d(ss);
  CombineTransposes pass;
  pass.run(&g);
  g.accept(&d);
  ASSERT_EQ("i_0.b_4.", ss.str());
  ASSERT_EQ(elw->getInput(0)->getNode()->getType(), mir::Operation::Type::input);
  ASSERT_EQ(elw->getInput(1)->getNode()->getType(), mir::Operation::Type::input);
}

TEST(OptPass, combineTransposesOpOrder)
{
  mir::Graph g;
  /*      Create graph:
   *   [input]     [input2]
   *      ||          ||
   * [Transpose 0] [Transpose1]
   *      ||          ||
   * [Transpose 2] [Transpose 3]
   *       \\       //
   *          [Add]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 2, 3}};
  Operation *in1 = g.create<ops::InputOp>(input_type);
  Operation *in2 = g.create<ops::InputOp>(input_type);
  Operation *tr0 = g.create<ops::TransposeOp>(in1->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *tr1 = g.create<ops::TransposeOp>(in2->getOutput(0), vector<size_t>{2, 1, 0});
  Operation *tr2 = g.create<ops::TransposeOp>(tr0->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *tr3 = g.create<ops::TransposeOp>(tr1->getOutput(0), vector<size_t>{2, 1, 0});
  Operation *elw = g.create<ops::AddOp>(tr2->getOutput(0), tr3->getOutput(0));
  g.create<ops::OutputOp>(elw->getOutput(0));
  int n1 = in1->getId();
  int n2 = in2->getId();
  CombineTransposes pass;
  pass.run(&g);
  ASSERT_EQ(g.getOutputs()[0]->getInput(0)->getNode()->getType(), mir::Operation::Type::add);
  // Order is preserved
  ASSERT_EQ(n1, elw->getInput(0)->getNode()->getId());
  ASSERT_EQ(n2, elw->getInput(1)->getNode()->getId());
}
} // unnamed namespace
