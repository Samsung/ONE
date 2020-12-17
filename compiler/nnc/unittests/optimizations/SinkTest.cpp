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

#include "passes/optimizations/SinkTranspose.h"
#include "passes/optimizations/SinkRelu.h"
#include "Util.h"
#include "mir/ops/TransposeOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/TanhOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/OutputOp.h"
#include "mir/Graph.h"

#include <gtest/gtest.h>
#include <sstream>
#include <vector>

using namespace std;
using namespace nnc;
using namespace mir;

namespace
{
Operation *getPrev(Operation *op)
{
  assert(op->getNumInputs() == 1);
  return op->getInput(0)->getNode();
}

Operation *getNext(Operation *op)
{
  assert(op->getNumOutputs() == 1 && (op->getOutput(0)->getUses().size() == 1));
  Operation::Use use = op->getOutput(0)->getUses()[0];
  return use.getNode();
}

/* This tests swapping relu and transpose */
TEST(OptPass, sinkTrReLU)
{
  mir::Graph g;
  /*
   * Create graph:
   *      [input]
   *        ||
   *    [Transpose]
   *        ||
   *      [relu]
   *        ||
   *      [tanh]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *tr1 = g.create<ops::TransposeOp>(input->getOutput(0), vector<size_t>{1, 0, 2});
  Operation *relu = g.create<ops::ReluOp>(tr1->getOutput(0));
  Operation *tanh = g.create<ops::TanhOp>(relu->getOutput(0));
  Operation *out = g.create<ops::OutputOp>(tanh->getOutput(0));
  (void)out;

  // Check that layout is desired
  SinkTranspose pass;
  pass.run(&g);

  // Assert transposes are removed
  ASSERT_EQ(g.getInputs()[0]->getType(), mir::Operation::Type::input);
  ASSERT_EQ(getPrev(g.getOutputs()[0])->getType(), mir::Operation::Type::tanh);
  ASSERT_EQ(getNext(g.getInputs()[0])->getType(), mir::Operation::Type::ReLU);
  ASSERT_EQ(getPrev(tanh)->getType(), mir::Operation::Type::transpose);
}

/* This tests swapping concat and transpose */
TEST(OptPass, sinkTrConcat)
{
  mir::Graph g;
  /*
   *   Create graph:
   *     [input]     [input2]
   *       ||          ||
   *  [Transpose 1] [Transpose 2]
   *         \\    //
   *         [Concat]
   *            ||
   *          [TanH]
   */

  mir::TensorType in1_type{mir::DataType::FLOAT32, Shape{1, 1, 2, 3}};
  Operation *in1 = g.create<ops::InputOp>(in1_type);

  mir::TensorType in2_type{mir::DataType::FLOAT32, Shape{1, 1, 2, 3}};
  Operation *in2 = g.create<ops::InputOp>(in2_type);
  Operation *tr1 = g.create<ops::TransposeOp>(in1->getOutput(0), vector<size_t>{0, 3, 1, 2});
  Operation *tr2 = g.create<ops::TransposeOp>(in2->getOutput(0), vector<size_t>{0, 3, 1, 2});
  Operation *conc =
    g.create<ops::ConcatOp>(vector<Operation::Output *>{tr1->getOutput(0), tr2->getOutput(0)}, 1);
  Operation *tanh = g.create<ops::TanhOp>(conc->getOutput(0));
  Operation *out = g.create<ops::OutputOp>(tanh->getOutput(0));
  (void)out;
  // Check that layout is as desired
  SinkTranspose pass;
  pass.run(&g);

  ASSERT_EQ(getPrev(getPrev(g.getOutputs()[0]))->getType(), Operation::Type::transpose);
  ASSERT_TRUE(static_cast<ops::TransposeOp *>(getPrev(tanh))->getAxisOrder() ==
              vector<size_t>({0, 3, 1, 2}));
  /* Expected Result:
   * TanH(Transpose(Concat(inp1,inp2)))
   */
}

/* This tests swapping concat and transpose */
TEST(OptPass, sinkReluConcat)
{
  mir::Graph g;
  /*
   *   Create graph:
   *     [ inp1 ]  [ inp2 ]
   *        ||        ||
   *     [ Relu 1] [ Relu 2]
   *         \\     //
   *        [ Concat ]
   *            ||
   *          [TanH]
   */
  mir::TensorType in1_type{mir::DataType::FLOAT32, Shape{1, 1, 2, 3}};
  Operation *in1 = g.create<ops::InputOp>(in1_type);

  mir::TensorType in2_type{mir::DataType::FLOAT32, Shape{1, 1, 2, 3}};
  Operation *in2 = g.create<ops::InputOp>(in2_type);
  Operation *relu1 = g.create<ops::ReluOp>(in1->getOutput(0));
  Operation *relu2 = g.create<ops::ReluOp>(in2->getOutput(0));
  Operation *conc = g.create<ops::ConcatOp>(
    vector<Operation::Output *>{relu1->getOutput(0), relu2->getOutput(0)}, 1);
  Operation *tanh = g.create<ops::TanhOp>(conc->getOutput(0));
  Operation *out = g.create<ops::OutputOp>(tanh->getOutput(0));
  (void)out;

  // Check that layout is as desired
  SinkRelu pass;
  pass.run(&g);

  ASSERT_EQ(getPrev(getPrev(g.getOutputs()[0]))->getType(), Operation::Type::ReLU);
  /* Expected Result:
   * TanH(Relu(Concat(inp1,inp2)))
   */
}

/* This tests swapping relu and max_pool */
TEST(OptPass, sinkPoolReLU)
{
  mir::Graph g;
  /*
   * Create graph:
   *      [input]
   *        ||
   *      [relu]
   *        ||
   *     [MaxPool]
   *        ||
   *      [tanh]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 4, 4, 3}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *relu = g.create<ops::ReluOp>(input->getOutput(0));
  mir::MaxPool2DOpAttributes attributes;
  attributes.window = {2, 2};
  attributes.strides = {2, 2};
  Operation *mp = g.create<ops::MaxPool2DOp>(relu->getOutput(0), attributes);
  Operation *tanh = g.create<ops::TanhOp>(mp->getOutput(0));
  Operation *out = g.create<ops::OutputOp>(tanh->getOutput(0));
  (void)out;

  SinkRelu pass;
  pass.run(&g);
  stringstream ss;
  DumpVisitor d{ss};
  g.accept(&d);

  // tanh(relu(pool(input)))
  ASSERT_EQ(getNext(g.getInputs()[0])->getType(), mir::Operation::Type::maxPool2D);
  ASSERT_EQ(getPrev(g.getOutputs()[0])->getType(), mir::Operation::Type::tanh);
  ASSERT_EQ("i_0.p_5.r_6.th_3.", ss.str());
}
} // unnamed namespace
