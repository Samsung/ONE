/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "InsertQuantizeOpOnDTypeMismatch.h"
#include "PassTestGraphs.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

std::unique_ptr<luci::CircleQuantParam> gen_qparam(float s, int64_t zp)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  {
    qparam->scale.push_back(s);
    qparam->zerop.push_back(zp);
  }

  return std::move(qparam);
}

/**
 *  Mul graph for test
 *
 * BEFORE
 *
 *   [Input(s16)]    [Const(s16)]
 *             \      /
 *            [Mul(u8)]
 *                |
 *           [Output(u8)]
 *
 * AFTER
 *
 *   [Input(s16)]    [Const(s16)]
 *             \      /
 *            [Mul(s16)]
 *                |
 *          [Quantize(u8)]
 *                |
 *           [Output(u8)]
 */
class MulGraphlet
{
public:
  MulGraphlet() = default;

  void init(loco::Graph *g)
  {
    _mul = g->nodes()->create<luci::CircleMul>();
    _const = g->nodes()->create<luci::CircleConst>();

    _mul->dtype(loco::DataType::U8);
    _const->dtype(loco::DataType::S16);

    _mul->quantparam(std::move(gen_qparam(1, 0)));
    _const->quantparam(std::move(gen_qparam(1, 0)));

    _mul->shape({2, 2, 2});

    _mul->fusedActivationFunction(luci::FusedActFunc::NONE);

    _mul->name("mul");
    _const->name("const");
  }

public:
  luci::CircleMul *mul(void) { return _mul; }

protected:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_const = nullptr;
};

class DtypeMisMatchMulTestGraph : public TestIOGraph, public MulGraphlet
{
public:
  void init(void)
  {
    TestIOGraph::init({2, 2, 2}, {2, 2, 2});

    input()->dtype(loco::DataType::S16);
    output()->dtype(loco::DataType::U8);

    input()->quantparam(std::move(gen_qparam(1, 0)));
    output()->quantparam(std::move(gen_qparam(1, 0)));

    MulGraphlet::init(g());

    _mul->x(input());
    _mul->y(_const);

    output()->from(_mul);
  }
};

} // namespace

TEST(InsertQuantizeOpOnDTypeMismatchTest, mul)
{
  DtypeMisMatchMulTestGraph g;

  luci::InsertQuantizeOpOnDTypeMismatch visitor;

  g.init();

  auto node = g.mul();
  node->accept(&visitor);

  // Quantize Op is created
  EXPECT_NE(nullptr, dynamic_cast<luci::CircleQuantize *>(g.output()->from()));

  // Mul's dtype is changed from U8 to S16
  EXPECT_EQ(loco::DataType::S16, g.mul()->dtype());
}

TEST(InsertQuantizeOpOnDTypeMismatchTest, mul_dtype_match_NEG)
{
  DtypeMisMatchMulTestGraph g;

  luci::InsertQuantizeOpOnDTypeMismatch visitor;

  g.init();

  auto node = g.mul();
  node->dtype(loco::DataType::S16);

  node->accept(&visitor);

  // Quantize Op is not created
  EXPECT_EQ(nullptr, dynamic_cast<luci::CircleQuantize *>(g.output()->from()));
}
