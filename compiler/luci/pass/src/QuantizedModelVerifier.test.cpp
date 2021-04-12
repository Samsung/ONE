/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizedModelVerifier.h"

#include "luci/Pass/QuantizeWithMinMaxPass.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using Type = loco::DataType;
using Granularity = luci::QuantizationGranularity;

namespace
{

/**
 * @brief A helper function to create dummy const node
 */
template <Type T> luci::CircleConst *create_dummy_const(loco::Graph *g, luci::test::ShapeU32 shape)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  {
    node->dtype(T);
    node->shape(shape);
    node->size<T>(luci::test::num_elements(shape));

    for (int32_t i = 0; i < luci::test::num_elements(shape); i++)
    {
      // DESIGN NOTE
      //
      // Filling with any random numbers are fine
      // Q. Should it include minus numbers?
      switch (T)
      {
        case Type::FLOAT32:
          // Fill with index
          node->at<T>(i) = static_cast<float>(i);
          break;
        case Type::BOOL:
          // Fill by flip
          node->at<T>(i) = (i % 2) ? true : false;
          break;
        case Type::U8:
          // Fill with index
          node->at<T>(i) = static_cast<uint8_t>(i);
          break;
        case Type::S16:
          // Fill with index
          node->at<T>(i) = static_cast<int16_t>(i);
          break;
      }
    }
  }

  return node;
}

void insert_scale_zp(luci::CircleNode *node, float scale, int64_t zp)
{
  auto qparam = node->quantparam();
  assert(qparam != nullptr); // FIX_CALLER_UNLESS
  qparam->scale.push_back(scale);
  qparam->zerop.push_back(zp);
}

void quantize_and_verify(loco::Graph *g, Type quantized_dtype, Granularity granularity)
{
  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, quantized_dtype, granularity);
  pass.run(g);

  luci::QuantizedModelVerifier verifier(quantized_dtype, granularity);
  verifier.verify(g);
}

// Helper function to reduce duplicate test codes
// Assumption: g->output()->from() is the target node
void quantize_and_verify_with_wrong_type(luci::test::TestIOGraph *g, Type quantized_dtype,
                                         Granularity granularity, Type wrong_dtype)
{
  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, quantized_dtype, granularity);
  pass.run(g->g());

  auto node = loco::must_cast<luci::CircleNode *>(g->output()->from());
  node->dtype(wrong_dtype);

  luci::QuantizedModelVerifier verifier(quantized_dtype, granularity);
  verifier.verify(g->g());
}

// Helper function to reduce duplicate test codes
// Assumption: g->output()->from() is the target node
void quantize_and_verify_with_wrong_granularity(luci::test::TestIOGraph *g, Type quantized_dtype,
                                                Granularity granularity)
{
  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, quantized_dtype, granularity);
  pass.run(g->g());

  auto node = loco::must_cast<luci::CircleNode *>(g->output()->from());
  insert_scale_zp(node, 1.0, 1);

  luci::QuantizedModelVerifier verifier(quantized_dtype, granularity);
  verifier.verify(g->g());
}

// Helper function to reduce duplicate test codes
void quantize_and_verify_with_wrong_granularity(luci::test::TestIOGraph *g, Type quantized_dtype,
                                                Granularity granularity, luci::CircleNode *target)
{
  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, quantized_dtype, granularity);
  pass.run(g->g());

  insert_scale_zp(target, 1.0, 1);

  luci::QuantizedModelVerifier verifier(quantized_dtype, granularity);
  verifier.verify(g->g());
}

// Set min/max for all non-const nodes in the graph
void set_minmax_to_non_const(loco::Graph *g, float min, float max)
{
  for (auto node : loco::all_nodes(g))
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (const_node != nullptr)
      continue;

    // Min/Max is not recorded for ArgMax
    // See MinMaxObserver.cpp in record_minmax module
    auto argmax_node = dynamic_cast<luci::CircleArgMax *>(node);
    if (argmax_node != nullptr)
      continue;

    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      qparam->min.emplace_back(min);
      qparam->max.emplace_back(max);
    }
    circle_node->quantparam(std::move(qparam));
  }
}

class LogisticTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _logistic = g()->nodes()->create<luci::CircleLogistic>();
    {
      _logistic->x(input());
    }
    output()->from(_logistic);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleLogistic *_logistic = nullptr;
};

class SoftmaxTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _softmax = g()->nodes()->create<luci::CircleSoftmax>();
    {
      _softmax->logits(input());
      _softmax->beta(0.1);
    }
    output()->from(_softmax);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleSoftmax *_softmax = nullptr;
};

template <Type indexT> class SliceTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _begin = g()->nodes()->create<luci::CircleConst>();
    {
      _begin->dtype(indexT);
    }
    _size = g()->nodes()->create<luci::CircleConst>();
    {
      _size->dtype(indexT);
    }
    _slice = g()->nodes()->create<luci::CircleSlice>();
    {
      _slice->input(input());
      _slice->begin(_begin);
      _slice->size(_size);
    }
    output()->from(_slice);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleSlice *_slice = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_size = nullptr;
};

class ReshapeTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _shape = g()->nodes()->create<luci::CircleConst>();
    {
      _shape->dtype(Type::S32);
    }
    _reshape = g()->nodes()->create<luci::CircleReshape>();
    {
      _reshape->tensor(input());
      _reshape->shape(_shape);
    }
    output()->from(_reshape);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleConst *_shape = nullptr;
};

class TanhTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _tanh = g()->nodes()->create<luci::CircleTanh>();
    {
      _tanh->x(input());
    }
    output()->from(_tanh);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleTanh *_tanh = nullptr;
};

class FloorTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _floor = g()->nodes()->create<luci::CircleFloor>();
    {
      _floor->x(input());
    }
    output()->from(_floor);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleFloor *_floor = nullptr;
};

template <Type indexT> class ArgMaxTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {1});
    // output dtype is float by default, but ArgMax should have indexType (s32/s64)
    output()->dtype(indexT);
    _dimension = g()->nodes()->create<luci::CircleConst>();
    {
      _dimension->dtype(indexT);
    }
    _argmax = g()->nodes()->create<luci::CircleArgMax>();
    {
      _argmax->input(input());
      _argmax->dimension(_dimension);
      _argmax->output_type(indexT);
      _argmax->dtype(indexT);
    }
    output()->from(_argmax);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleArgMax *_argmax = nullptr;
  luci::CircleConst *_dimension = nullptr;
};

class PadTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _paddings = g()->nodes()->create<luci::CircleConst>();
    {
      _paddings->dtype(Type::S32);
    }
    _pad = g()->nodes()->create<luci::CirclePad>();
    {
      _pad->input(input());
      _pad->paddings(_paddings);
    }
    output()->from(_pad);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CirclePad *_pad = nullptr;
  luci::CircleConst *_paddings = nullptr;
};

class TransposeTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _perm = g()->nodes()->create<luci::CircleConst>();
    {
      _perm->dtype(Type::S32);
    }
    _transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _transpose->a(input());
      _transpose->perm(_perm);
    }
    output()->from(_transpose);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleTranspose *_transpose = nullptr;
  luci::CircleConst *_perm = nullptr;
};

class ConcatenationTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({16}, {32});
    
    _param = create_dummy_const<Type::FLOAT32>(g(), {16});
    _concat = g()->nodes()->create<luci::CircleConcatenation>(2);
    {
      _concat->values(0, input());
      _concat->values(1, _param);
      _concat->axis(0);
    }
    output()->from(_concat);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleConcatenation *_concat = nullptr;
  luci::CircleConst *_param = nullptr;
};

// Test graph for comparison Ops
// GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, NOT_EQUAL
template <class Op> class ComparisonOpTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    output()->dtype(loco::DataType::BOOL);
    _y = create_dummy_const<Type::FLOAT32>(g(), {32});
    _op = g()->nodes()->create<Op>();
    {
      _op->x(input());
      _op->y(_y);
      _op->dtype(loco::DataType::BOOL);
    }
    output()->from(_op);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _op->x(); }
  loco::Node *y() { return _op->y(); }

public:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

// Test graph for binary logical Ops
// LOGICAL_OR, LOGICAL_AND
template <class Op> class BinaryLogicalOpTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    input()->dtype(loco::DataType::BOOL);
    output()->dtype(loco::DataType::BOOL);
    _y = create_dummy_const<Type::BOOL>(g(), {32});
    _op = g()->nodes()->create<Op>();
    {
      _op->x(input());
      _op->y(_y);
      _op->dtype(loco::DataType::BOOL);
    }
    output()->from(_op);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _op->x(); }
  loco::Node *y() { return _op->y(); }

public:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

class DivTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _div = g()->nodes()->create<luci::CircleDiv>();
    {
      _div->x(input());
      _div->y(_const);
    }
    output()->from(_div);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _div->x(); }
  loco::Node *y() { return _div->y(); }

private:
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_const = nullptr;
};

class FloorDivTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _floor_div = g()->nodes()->create<luci::CircleFloorDiv>();
    {
      _floor_div->x(input());
      _floor_div->y(_const);
    }
    output()->from(_floor_div);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _floor_div->x(); }

  loco::Node *y() { return _floor_div->y(); }

private:
  luci::CircleFloorDiv *_floor_div = nullptr;
  luci::CircleConst *_const = nullptr;
};

class RsqrtTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _rsqrt = g()->nodes()->create<luci::CircleRsqrt>();
    {
      _rsqrt->x(input());
    }
    output()->from(_rsqrt);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  luci::CircleRsqrt *_rsqrt = nullptr;
};

} // namespace

// Quantize and verify with given configurations
#define TEST_WITH_GRAPH(graph, type, granularity)                   \
  do                                                                \
  {                                                                 \
    graph g;                                                        \
    g.init();                                                       \
    EXPECT_NO_THROW(quantize_and_verify(g.g(), type, granularity)); \
  } while (0)

// Quantize and verify with wrong type
#define TEST_WITH_WRONG_TYPE(graph, type, granularity, wrong_dtype)                            \
  do                                                                                           \
  {                                                                                            \
    graph g;                                                                                   \
    g.init();                                                                                  \
    EXPECT_ANY_THROW(quantize_and_verify_with_wrong_type(&g, type, granularity, wrong_dtype)); \
  } while (0)

// Quantize and verify with wrong granularity
#define TEST_WITH_WRONG_GRANULARITY(graph, type, granularity)                            \
  do                                                                                     \
  {                                                                                      \
    graph g;                                                                             \
    g.init();                                                                            \
    EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&g, type, granularity)); \
  } while (0)

// Quantize and verify with wrong granularity
// Users can specify the test target
#define TEST_WITH_WRONG_GRANULARITY_TARGET(graph, type, granularity, target)                   \
  do                                                                                           \
  {                                                                                            \
    graph g;                                                                                   \
    g.init();                                                                                  \
    auto node = loco::must_cast<luci::CircleNode *>(target);                                   \
    EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&g, type, granularity, node)); \
  } while (0)

// Test a local helper function
TEST(QuantizedModelVerifierTest, LocalCreateDummyConst)
{
  loco::Graph g;

  EXPECT_NO_THROW(create_dummy_const<Type::FLOAT32>(&g, {32, 32}));
}

TEST(QuantizedModelVerifierTest, Logistic)
{
  TEST_WITH_GRAPH(LogisticTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(LogisticTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(LogisticTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Softmax)
{
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Softmax_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Softmax_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Slice)
{
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Slice_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise, Type::U8);

  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Slice_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, ArgMax)
{
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, ArgMax_wrong_dimension_type_NEG)
{
  ArgMaxTestGraph<Type::S32> g;
  g.init();
  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
  pass.run(g.g());

  g._dimension->dtype(Type::U8);

  luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
  EXPECT_ANY_THROW(verifier.verify(g.g()));
}

TEST(QuantizedModelVerifierTest, ArgMax_wrong_input_granularity_NEG)
{
  ArgMaxTestGraph<Type::S32> g;
  g.init();

  luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
  pass.run(g.g());

  insert_scale_zp(loco::must_cast<luci::CircleNode *>(g._argmax->input()), 1.0, 1);

  luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
  EXPECT_ANY_THROW(verifier.verify(g.g()));
}

TEST(QuantizedModelVerifierTest, Concatenation)
{
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Concatenation_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Concatenation_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, LogicalOr)
{
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                  Granularity::LayerWise);
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                  Granularity::ChannelWise);
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::S16,
                  Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, LogicalOr_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
}

TEST(QuantizedModelVerifierTest, Reshape)
{
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Reshape_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Reshape_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Tanh)
{
  TEST_WITH_GRAPH(TanhTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(TanhTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(TanhTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Tanh_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Tanh_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Pad)
{
  TEST_WITH_GRAPH(PadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(PadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(PadTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Pad_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Pad_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Transpose)
{
  TEST_WITH_GRAPH(TransposeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(TransposeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(TransposeTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Transpose_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Transpose_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Floor)
{
  TEST_WITH_GRAPH(FloorTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(FloorTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(FloorTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Floor_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Floor_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, GreaterEqual)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                  Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                  Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                  Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g.y());
}

TEST(QuantizedModelVerifierTest, Greater)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Greater_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::LayerWise,
                       Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
}

TEST(QuantizedModelVerifierTest, Greater_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g.y());
}

TEST(QuantizedModelVerifierTest, NotEqual)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g.y());
}

TEST(QuantizedModelVerifierTest, Div)
{
  TEST_WITH_GRAPH(DivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(DivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(DivTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Div_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Div_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::S16, Granularity::ChannelWise, g.y());
}

TEST(QuantizedModelVerifierTest, FloorDiv)
{
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, g.y());
}

TEST(QuantizedModelVerifierTest, Rsqrt)
{
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::S16, Granularity::ChannelWise);
}

TEST(QuantizedModelVerifierTest, Rsqrt_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
}

TEST(QuantizedModelVerifierTest, Rsqrt_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::S16, Granularity::ChannelWise);
}

#undef TEST_WITH_GRAPH
#undef TEST_WITH_WRONG_TYPE
#undef TEST_WITH_WRONG_GRANULARITY
