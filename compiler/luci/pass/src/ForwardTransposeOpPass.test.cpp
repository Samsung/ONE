/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ForwardTransposeOpPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <logo/Phase.h>
#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

#include <vector>

namespace
{

using namespace luci::test;

template <typename T> class TransposeBinaryOpGraphlet
{
public:
  TransposeBinaryOpGraphlet() = default;

public:
  virtual ~TransposeBinaryOpGraphlet() = default;

public:
  // TODO Rename shape_in to shape_const
  void init(loco::Graph *g, const ShapeU32 shape_in, const ShapeU32 perm)
  {
    std::vector<uint32_t> shape_in_v = shape_in;
    std::vector<uint32_t> perm_v = perm;

    _perm = g->nodes()->create<luci::CircleConst>();
    _const = g->nodes()->create<luci::CircleConst>();
    _transpose = g->nodes()->create<luci::CircleTranspose>();
    _binary = g->nodes()->create<T>();

    _perm->dtype(loco::DataType::S32);
    _perm->rank(1);
    _perm->dim(0).set(perm_v.size());
    _perm->shape_status(luci::ShapeStatus::VALID);

    _const->dtype(loco::DataType::FLOAT32);
    _const->rank(shape_in_v.size());
    for (uint32_t i = 0; i < shape_in_v.size(); i++)
      _const->dim(i).set(shape_in_v[perm_v[i]]);
    _const->shape_status(luci::ShapeStatus::VALID);

    // values
    const auto size = perm_v.size();
    _perm->size<loco::DataType::S32>(size);
    for (uint32_t i = 0; i < size; i++)
      _perm->at<loco::DataType::S32>(i) = perm_v[i];

    uint32_t elems = 1;
    for (uint32_t i = 0; i < shape_in_v.size(); i++)
      elems *= shape_in_v[i];

    _const->size<loco::DataType::FLOAT32>(elems);
    for (uint32_t i = 0; i < elems; i++)
      _const->at<loco::DataType::FLOAT32>(i) = i;

    _perm->name("transpose_perm");
    _transpose->name("transpose");
    _binary->name("binary");
  }

  luci::CircleTranspose *transpose(void) { return _transpose; }

  void switch_xy(void)
  {
    assert(_binary); // FIX_CALLER_UNLESS
    auto temp = _binary->x();
    _binary->x(_binary->y());
    _binary->y(temp);
  }

protected:
  luci::CircleTranspose *_transpose = nullptr;
  T *_binary = nullptr;
  luci::CircleConst *_perm = nullptr;
  luci::CircleConst *_const = nullptr;
};

using TransposeAddGraphlet = TransposeBinaryOpGraphlet<luci::CircleAdd>;
using TransposeMulGraphlet = TransposeBinaryOpGraphlet<luci::CircleMul>;

class ForwardTransposeToAddGraph : public TestIOGraph, public TransposeAddGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeAddGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _binary->x(_transpose);
    _binary->y(_const);

    output()->from(_binary);
  }
};

class ForwardTransposeToAddInvalidGraph : public TestIOGraph, public TransposeAddGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeAddGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _binary->x(_transpose);
    _binary->y(_transpose);

    output()->from(_binary);
  }
};

class ForwardTransposeToMulGraph : public TestIOGraph, public TransposeMulGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeMulGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _binary->x(_transpose);
    _binary->y(_const);

    output()->from(_binary);
  }
};

class ForwardTransposeToScalarMulGraph : public TestIOGraph, public TransposeMulGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeMulGraphlet::init(g(), {}, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _binary->x(_transpose);
    _binary->y(_const);

    output()->from(_binary);
  }
};

class ForwardTransposeToSingleElemMulGraph : public TestIOGraph, public TransposeMulGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeMulGraphlet::init(g(), {1}, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _binary->x(_transpose);
    _binary->y(_const);

    output()->from(_binary);
  }
};

void run_phase(loco::Graph *g)
{
  logo::Phase phase;

  // Default passes.
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

  // Pass to test
  phase.emplace_back(std::make_unique<luci::ForwardTransposeOpPass>());

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

class ForwardTransposeToAddGraphTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToAddGraph _graph;
};

class ForwardTransposeToAddGraphNegTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToAddInvalidGraph _graph;
};

class ForwardTransposeToMulGraphTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToMulGraph _graph;
};

class ForwardTransposeToScalarMulGraphTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToScalarMulGraph _graph;
};

class ForwardTransposeToSingleElemMulGraphTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToSingleElemMulGraph _graph;
};

} // namespace

TEST_F(ForwardTransposeToAddGraphTest, forward_add_xy)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto add = dynamic_cast<luci::CircleAdd *>(transpose->a());
  EXPECT_NE(nullptr, add);
  EXPECT_EQ(4, add->rank());
  EXPECT_EQ(1, add->dim(0).value());
  EXPECT_EQ(64, add->dim(1).value());
  EXPECT_EQ(51, add->dim(2).value());
  EXPECT_EQ(1, add->dim(3).value());

  auto add_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, add_const);
  EXPECT_EQ(4, add_const->rank());
  EXPECT_EQ(1, add_const->dim(0).value());
  EXPECT_EQ(64, add_const->dim(1).value());
  EXPECT_EQ(51, add_const->dim(2).value());
  EXPECT_EQ(1, add_const->dim(3).value());
}

TEST_F(ForwardTransposeToAddGraphTest, forward_add_yx)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});
  _graph.switch_xy();

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto mul = dynamic_cast<luci::CircleAdd *>(transpose->a());
  EXPECT_NE(nullptr, mul);
  EXPECT_EQ(4, mul->rank());
  EXPECT_EQ(1, mul->dim(0).value());
  EXPECT_EQ(64, mul->dim(1).value());
  EXPECT_EQ(51, mul->dim(2).value());
  EXPECT_EQ(1, mul->dim(3).value());

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->x());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_EQ(4, mul_const->rank());
  EXPECT_EQ(1, mul_const->dim(0).value());
  EXPECT_EQ(64, mul_const->dim(1).value());
  EXPECT_EQ(51, mul_const->dim(2).value());
  EXPECT_EQ(1, mul_const->dim(3).value());
}

TEST_F(ForwardTransposeToMulGraphTest, forward_mul_xy)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto mul = dynamic_cast<luci::CircleMul *>(transpose->a());
  EXPECT_NE(nullptr, mul);
  EXPECT_EQ(4, mul->rank());
  EXPECT_EQ(1, mul->dim(0).value());
  EXPECT_EQ(64, mul->dim(1).value());
  EXPECT_EQ(51, mul->dim(2).value());
  EXPECT_EQ(1, mul->dim(3).value());

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_EQ(4, mul_const->rank());
  EXPECT_EQ(1, mul_const->dim(0).value());
  EXPECT_EQ(64, mul_const->dim(1).value());
  EXPECT_EQ(51, mul_const->dim(2).value());
  EXPECT_EQ(1, mul_const->dim(3).value());
}

TEST_F(ForwardTransposeToMulGraphTest, forward_mul_yx)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});
  _graph.switch_xy();

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto mul = dynamic_cast<luci::CircleMul *>(transpose->a());
  EXPECT_NE(nullptr, mul);
  EXPECT_EQ(4, mul->rank());
  EXPECT_EQ(1, mul->dim(0).value());
  EXPECT_EQ(64, mul->dim(1).value());
  EXPECT_EQ(51, mul->dim(2).value());
  EXPECT_EQ(1, mul->dim(3).value());

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->x());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_EQ(4, mul_const->rank());
  EXPECT_EQ(1, mul_const->dim(0).value());
  EXPECT_EQ(64, mul_const->dim(1).value());
  EXPECT_EQ(51, mul_const->dim(2).value());
  EXPECT_EQ(1, mul_const->dim(3).value());
}

TEST_F(ForwardTransposeToScalarMulGraphTest, forward_scalar_mul)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto mul = dynamic_cast<luci::CircleMul *>(transpose->a());
  EXPECT_NE(nullptr, mul);
  EXPECT_EQ(4, mul->rank());
  EXPECT_EQ(1, mul->dim(0).value());
  EXPECT_EQ(64, mul->dim(1).value());
  EXPECT_EQ(51, mul->dim(2).value());
  EXPECT_EQ(1, mul->dim(3).value());

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_EQ(0, mul_const->rank());
}

TEST_F(ForwardTransposeToSingleElemMulGraphTest, forward_single_elem_mul)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto mul = dynamic_cast<luci::CircleMul *>(transpose->a());
  EXPECT_NE(nullptr, mul);
  EXPECT_EQ(4, mul->rank());
  EXPECT_EQ(1, mul->dim(0).value());
  EXPECT_EQ(64, mul->dim(1).value());
  EXPECT_EQ(51, mul->dim(2).value());
  EXPECT_EQ(1, mul->dim(3).value());

  auto mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
  EXPECT_NE(nullptr, mul_const);
  EXPECT_EQ(1, mul_const->rank());
  EXPECT_EQ(1, mul_const->dim(0).value());
}

TEST_F(ForwardTransposeToAddGraphTest, forward_transpose_add_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  // Remove add
  _graph.output()->from(_graph.transpose());

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

TEST_F(ForwardTransposeToAddGraphNegTest, forward_transpose_add_non_const_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

TEST_F(ForwardTransposeToMulGraphTest, forward_transpose_mul_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  // Remove mul
  _graph.output()->from(_graph.transpose());

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

// Unary

namespace
{

template <typename T> class TransposeUnaryOpGraphlet
{
public:
  TransposeUnaryOpGraphlet() = default;

public:
  virtual ~TransposeUnaryOpGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 shape_in, const ShapeU32 perm)
  {
    std::vector<uint32_t> shape_in_v = shape_in;
    std::vector<uint32_t> perm_v = perm;

    assert(shape_in_v.size() == perm_v.size()); // FIX_CALLER_UNLESS

    _perm = g->nodes()->create<luci::CircleConst>();
    _const = g->nodes()->create<luci::CircleConst>();
    _transpose = g->nodes()->create<luci::CircleTranspose>();
    _unary = g->nodes()->create<T>();

    _perm->dtype(loco::DataType::S32);
    _perm->rank(1);
    _perm->dim(0).set(perm_v.size());
    _perm->shape_status(luci::ShapeStatus::VALID);

    _const->dtype(loco::DataType::FLOAT32);
    _const->rank(shape_in_v.size());
    for (uint32_t i = 0; i < shape_in_v.size(); i++)
      _const->dim(i).set(shape_in_v[perm_v[i]]);
    _const->shape_status(luci::ShapeStatus::VALID);

    // values
    const auto size = perm_v.size();
    _perm->size<loco::DataType::S32>(size);
    for (uint32_t i = 0; i < size; i++)
      _perm->at<loco::DataType::S32>(i) = perm_v[i];

    uint32_t elems = 1;
    for (uint32_t i = 0; i < size; i++)
      elems *= shape_in_v[i];

    _const->size<loco::DataType::FLOAT32>(elems);
    for (uint32_t i = 0; i < elems; i++)
      _const->at<loco::DataType::FLOAT32>(i) = i;

    _perm->name("transpose_perm");
    _transpose->name("transpose");
    _unary->name("_unary");
  }

  luci::CircleTranspose *transpose(void) { return _transpose; }

protected:
  luci::CircleTranspose *_transpose = nullptr;
  T *_unary = nullptr;
  luci::CircleConst *_perm = nullptr;
  luci::CircleConst *_const = nullptr;
};

using TransposeAbsGraphlet = TransposeUnaryOpGraphlet<luci::CircleAbs>;

class ForwardTransposeToAbsGraph : public TestIOGraph, public TransposeAbsGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeAbsGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _transpose->a(input());
    _transpose->perm(_perm);
    _unary->x(_transpose);

    output()->from(_unary);
  }
};

class ForwardTransposeToAbsInvalidGraph : public TestIOGraph, public TransposeAbsGraphlet
{
public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    TransposeAbsGraphlet::init(g(), shape_in, shape_out);

    _relu = g()->nodes()->create<luci::CircleRelu>();
    _relu->dtype(loco::DataType::FLOAT32);
    _relu->name("relu");

    // connect network
    _relu->features(input());
    _unary->x(_relu);

    output()->from(_unary);
  }

protected:
  luci::CircleRelu *_relu = nullptr;
};

class ForwardTransposeToAbsGraphTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToAbsGraph _graph;
};

class ForwardTransposeToAbsGraphNegTest : public ::testing::Test
{
public:
  void run_pass(void) { run_phase(_graph.g()); }

protected:
  ForwardTransposeToAbsInvalidGraph _graph;
};

} // namespace

TEST_F(ForwardTransposeToAbsGraphTest, forward_abs_x)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  run_pass();

  auto transpose = dynamic_cast<luci::CircleTranspose *>(_graph.output()->from());
  EXPECT_NE(nullptr, transpose);
  EXPECT_EQ(4, transpose->rank());
  EXPECT_EQ(1, transpose->dim(0).value());
  EXPECT_EQ(1, transpose->dim(1).value());
  EXPECT_EQ(51, transpose->dim(2).value());
  EXPECT_EQ(64, transpose->dim(3).value());

  auto abs = dynamic_cast<luci::CircleAbs *>(transpose->a());
  EXPECT_NE(nullptr, abs);
  EXPECT_EQ(4, abs->rank());
  EXPECT_EQ(1, abs->dim(0).value());
  EXPECT_EQ(64, abs->dim(1).value());
  EXPECT_EQ(51, abs->dim(2).value());
  EXPECT_EQ(1, abs->dim(3).value());
}

TEST_F(ForwardTransposeToAbsGraphTest, forward_transpose_abs_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  // Remove abs
  _graph.output()->from(_graph.transpose());

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

TEST_F(ForwardTransposeToAbsGraphNegTest, forward_transpose_abs_non_transpose_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

TEST_F(ForwardTransposeToScalarMulGraphTest, forward_transpose_smul_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  // Remove mul
  _graph.output()->from(_graph.transpose());

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}

TEST_F(ForwardTransposeToSingleElemMulGraphTest, forward_transpose_se_mul_NEG)
{
  _graph.init({1, 64, 51, 1}, {0, 3, 2, 1});

  // Remove mul
  _graph.output()->from(_graph.transpose());

  luci::ForwardTransposeOpPass pass;
  EXPECT_FALSE(pass.run(_graph.g()));
}
