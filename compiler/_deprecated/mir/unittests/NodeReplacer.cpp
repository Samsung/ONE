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

#include <gtest/gtest.h>

#include "mir/Graph.h"
#include "mir/Visitor.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/ReluOp.h"

namespace
{

using namespace mir;

class DumpVisitor : public Visitor
{
public:
  DumpVisitor(std::ostream &s) : _s(s) {}

  void visit(ops::InputOp &op) override { _s << "i" << std::to_string(op.getId()); };

  void visit(ops::ReluOp &op) override { _s << "r" << std::to_string(op.getId()); }

  void visit(ops::ConcatOp &op) override { _s << "c" << std::to_string(op.getId()); }

  std::ostream &_s;
};

TEST(NodeMutatorTest, SimpleChainTest)
{
  auto g = new Graph;
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{}};
  auto n1 = g->create<ops::InputOp>(input_type);
  auto n2 = g->create<ops::ReluOp>(n1->getOutput(0));
  auto n3 = g->create<ops::ReluOp>(n2->getOutput(0));
  auto n4 = g->create<ops::ReluOp>(n2->getOutput(0));
  auto n5 = g->create<ops::ReluOp>(n1->getOutput(0));

  g->replaceNode(n2, n5);

  std::stringstream ss;
  DumpVisitor d(ss);
  g->accept(&d);

  auto str = ss.str();
  ASSERT_TRUE(str == "i0r4r2r3" || str == "i0r4r3r2") << "str = " << str;
  delete g;
}

} // namespace
