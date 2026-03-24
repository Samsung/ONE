/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelAnalyzer.h"
#include "mir/Graph.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ConcatOp.h"

#include <gtest/gtest.h>

#include <algorithm>

using namespace std;
using namespace nnc;
using namespace mir;
using namespace sir;

static const CallFunction *getCall(const unique_ptr<Action> &ptr)
{
  return dynamic_cast<const CallFunction *>(ptr.get());
}

/*
 * This test designed to check basic layout properties of Model analyzer
 */
TEST(ModelAnalyzer, linearization)
{
  mir::Graph g;
  /*
   * Create graph:
   *      [input]
   *     /       \
   *    |         |
   *    V         V
   * [head1]   [head2]
   *    |         |
   *    V         V
   * [tail1]   [tail2]
   *     \       /
   *      \     /
   *      [join]
   */
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3}};
  Operation *input = g.create<ops::InputOp>(input_type);
  Operation *head1 = g.create<ops::ReluOp>(input->getOutput(0));
  Operation *head2 = g.create<ops::ReluOp>(input->getOutput(0));
  Operation *tail1 = g.create<ops::ReluOp>(head1->getOutput(0));
  Operation *tail2 = g.create<ops::ReluOp>(head2->getOutput(0));
  vector<mir::Operation::Output *> concat_inputs{tail1->getOutput(0), tail2->getOutput(0)};
  Operation *join = g.create<ops::ConcatOp>(concat_inputs, 0);
  input->getOutput(0)->setName("input");
  head1->getOutput(0)->setName("head1");
  head2->getOutput(0)->setName("head2");
  tail1->getOutput(0)->setName("tail2");
  tail2->getOutput(0)->setName("tail2");
  join->getOutput(0)->setName("join");

  // Check that layout is desired
  ModelAnalyzer ma;
  ma.analyze(&g);
  const auto &seq = ma.getInferenceSequence();
  ASSERT_EQ(seq.size(), 6u);

  vector<Operation *> op_seq(seq.size());
  transform(seq.cbegin(), seq.cend(), op_seq.begin(),
            [](const unique_ptr<sir::Action> &action) { return getCall(action)->mirOp; });

  vector<Operation *> valid_seq1{input, head1, tail1, head2, tail2, join};
  vector<Operation *> valid_seq2{input, head2, tail2, head1, tail1, join};
  ASSERT_TRUE(op_seq == valid_seq1 || op_seq == valid_seq2);
}
