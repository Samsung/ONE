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
#include "luci/IR/CircleOpcode.h"

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

class SingleOpTestGraph : public luci::test::TestIOGraph
{
public:
  virtual void init(void) = 0;
};

class LogisticTestGraph final : public SingleOpTestGraph
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

class SoftmaxTestGraph final : public SingleOpTestGraph
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

template <Type indexT> class SliceTestGraph final : public SingleOpTestGraph
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

class ReshapeTestGraph final : public SingleOpTestGraph
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

class TanhTestGraph final : public SingleOpTestGraph
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

class FloorTestGraph final : public SingleOpTestGraph
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

template <Type indexT> class ArgMaxTestGraph final : public SingleOpTestGraph
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

class PadTestGraph final : public SingleOpTestGraph
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

class TransposeTestGraph final : public SingleOpTestGraph
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

class ConcatenationTestGraph final : public SingleOpTestGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({16}, {32});
    _param = g()->nodes()->create<luci::CircleConst>();
    {
      _param->dtype(Type::FLOAT32);
      _param->shape({16});
      _param->size<Type::FLOAT32>(16);
      for (int16_t i = 0; i < 16; i++)
        _param->at<Type::FLOAT32>(i) = static_cast<float>(i);
    }
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
template <class Op> class ComparisonOpTestGraph final : public SingleOpTestGraph
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

public:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

// Test graph for binary logical Ops
// LOGICAL_OR, LOGICAL_AND
template <class Op> class BinaryLogicalOpTestGraph final : public SingleOpTestGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    input()->dtype(loco::DataType::BOOL);
    output()->dtype(loco::DataType::BOOL);
    _y = g()->nodes()->create<luci::CircleConst>();
    {
      _y->dtype(Type::BOOL);
      _y->shape({32});
      _y->size<Type::BOOL>(32);
      for (int32_t i = 0; i < 32; i++)
        _y->at<Type::BOOL>(i) = 0;
    }
    _op = g()->nodes()->create<Op>();
    {
      _op->x(input());
      _op->y(_y);
      _op->dtype(loco::DataType::BOOL);
    }
    output()->from(_op);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

class DivTestGraph final : public SingleOpTestGraph
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

class FloorDivTestGraph final : public SingleOpTestGraph
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

} // namespace

// TODO : remove
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

// Quantize the given graph with given configuration
template <typename Graph> void test_with_graph(Type type, Granularity granularity)
{
  Graph graph;
  graph.init();

  assert(!(type == Type::S16 && granularity == Granularity::LayerWise));

  EXPECT_NO_THROW(quantize_and_verify(graph.g(), type, granularity));
}

// Quantize the given graph with given configuration
// Change the type of output node to be wrong and expect the verifier to throw exception
template <typename Graph> void test_with_wrong_type(Type type, Granularity granularity)
{
  Graph graph;
  graph.init();

  Type wrong_type = (type == Type::U8) ? Type::S16 : Type::U8;

  EXPECT_ANY_THROW(quantize_and_verify_with_wrong_type(&graph, type, granularity, wrong_type));
}

// Quantize the given graph with given configuration
// Change the granularity of output node to be wrong and expect the verifier to throw exception
template <typename Graph> void test_with_wrong_granularity(Type type, Granularity granularity)
{
  Graph graph;
  graph.init();

  EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&graph, type, granularity));
}

template <typename Graph>
void test_with_wrong_granularity_with_target_x(Type type, Granularity granularity)
{
  Graph graph;
  graph.init();
  auto node = loco::must_cast<luci::CircleNode *>(graph.x());

  EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&graph, type, granularity, node));
}

template <typename Graph>
void test_with_wrong_granularity_with_target_y(Type type, Granularity granularity)
{
  Graph graph;
  graph.init();
  auto node = loco::must_cast<luci::CircleNode *>(graph.y());

  EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&graph, type, granularity, node));
}
typedef struct
{
  Type type;
  Granularity granularity;
} attr;

std::initializer_list<attr> attr_list = {
  {Type::U8, Granularity::LayerWise},
  {Type::U8, Granularity::ChannelWise},
  {Type::S16, Granularity::ChannelWise},
  // {Type::S16, Granularity::ChannelWise} is not supported
};

template <typename Graph> void test_with_graph()
{
  for (auto attr : attr_list)
  {
    test_with_graph<Graph>(attr.type, attr.granularity);
  }
}

template <typename Graph> void test_with_wrong_type()
{
  for (auto attr : attr_list)
  {
    test_with_wrong_type<Graph>(attr.type, attr.granularity);
  }
}

template <typename Graph> void test_with_wrong_granularity()
{
  for (auto attr : attr_list)
  {
    test_with_wrong_granularity<Graph>(attr.type, attr.granularity);
  }
}

template <typename Graph> void test_with_wrong_granularity_with_target_x()
{
  for (auto attr : attr_list)
  {
    test_with_wrong_granularity_with_target_x<Graph>(attr.type, attr.granularity);
  }
}

template <typename Graph> void test_with_wrong_granularity_with_target_y()
{
  for (auto attr : attr_list)
  {
    test_with_wrong_granularity_with_target_x<Graph>(attr.type, attr.granularity);
  }
}

// Test a local helper function
TEST(QuantizedModelVerifierTest, LocalCreateDummyConst)
{
  loco::Graph g;

  EXPECT_NO_THROW(create_dummy_const<Type::FLOAT32>(&g, {32, 32}));
}

TEST(QuantizedModelVerifierTest, Logistic) { test_with_graph<LogisticTestGraph>(); }

TEST(QuantizedModelVerifierTest, Logistic_wrong_type_NEG)
{
  test_with_wrong_type<LogisticTestGraph>();
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_granularity_NEG)
{
  test_with_wrong_granularity<LogisticTestGraph>();
}

TEST(QuantizedModelVerifierTest, Softmax) { test_with_graph<SoftmaxTestGraph>(); }

TEST(QuantizedModelVerifierTest, Softmax_wrong_type_NEG)
{
  test_with_wrong_type<SoftmaxTestGraph>();
}

TEST(QuantizedModelVerifierTest, Softmax_wrong_granularity_NEG)
{
  test_with_wrong_granularity<SoftmaxTestGraph>();
}

TEST(QuantizedModelVerifierTest, Slice)
{
  test_with_graph<SliceTestGraph<Type::S32>>();
  test_with_graph<SliceTestGraph<Type::S64>>();
}

TEST(QuantizedModelVerifierTest, Slice_wrong_type_NEG)
{
  test_with_wrong_type<SliceTestGraph<Type::S32>>();
  test_with_wrong_type<SliceTestGraph<Type::S64>>();
}

TEST(QuantizedModelVerifierTest, Slice_wrong_granularity_NEG)
{
  test_with_wrong_granularity<SliceTestGraph<Type::S32>>();
  test_with_wrong_granularity<SliceTestGraph<Type::S64>>();
}

TEST(QuantizedModelVerifierTest, ArgMax)
{
  test_with_graph<ArgMaxTestGraph<Type::S32>>();
  test_with_graph<ArgMaxTestGraph<Type::S64>>();
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

TEST(QuantizedModelVerifierTest, Concatenation) { test_with_graph<ConcatenationTestGraph>(); }

TEST(QuantizedModelVerifierTest, Concatenation_wrong_type_NEG)
{
  test_with_wrong_type<ConcatenationTestGraph>();
}

TEST(QuantizedModelVerifierTest, Concatenation_wrong_granularity_NEG)
{
  test_with_wrong_granularity<ConcatenationTestGraph>();
}

TEST(QuantizedModelVerifierTest, LogicalOr)
{
  test_with_graph<BinaryLogicalOpTestGraph<luci::CircleLogicalOr>>();
}

TEST(QuantizedModelVerifierTest, LogicalOr_wrong_type_NEG)
{
  test_with_wrong_type<BinaryLogicalOpTestGraph<luci::CircleLogicalOr>>();
}

TEST(QuantizedModelVerifierTest, Reshape) { test_with_graph<ReshapeTestGraph>(); }

TEST(QuantizedModelVerifierTest, Reshape_wrong_type_NEG)
{
  test_with_wrong_type<ReshapeTestGraph>();
}

TEST(QuantizedModelVerifierTest, Reshape_wrong_granularity_NEG)
{
  test_with_wrong_granularity<ReshapeTestGraph>();
}

TEST(QuantizedModelVerifierTest, Tanh) { test_with_graph<TanhTestGraph>(); }

TEST(QuantizedModelVerifierTest, Tanh_wrong_type_NEG) { test_with_wrong_type<TanhTestGraph>(); }

TEST(QuantizedModelVerifierTest, Tanh_wrong_granularity_NEG)
{
  test_with_wrong_granularity<TanhTestGraph>();
}

TEST(QuantizedModelVerifierTest, Pad) { test_with_graph<PadTestGraph>(); }

TEST(QuantizedModelVerifierTest, Pad_wrong_type_NEG) { test_with_wrong_type<PadTestGraph>(); }

TEST(QuantizedModelVerifierTest, Pad_wrong_granularity_NEG)
{
  test_with_wrong_granularity<PadTestGraph>();
}

TEST(QuantizedModelVerifierTest, Transpose) { test_with_graph<TransposeTestGraph>(); }

TEST(QuantizedModelVerifierTest, Transpose_wrong_type_NEG)
{
  test_with_wrong_type<TransposeTestGraph>();
}

TEST(QuantizedModelVerifierTest, Transpose_wrong_granularity_NEG)
{
  test_with_wrong_granularity<TransposeTestGraph>();
}

TEST(QuantizedModelVerifierTest, Floor) { test_with_graph<FloorTestGraph>(); }

TEST(QuantizedModelVerifierTest, Floor_wrong_type_NEG) { test_with_wrong_type<FloorTestGraph>(); }

TEST(QuantizedModelVerifierTest, Floor_wrong_granularity_NEG)
{
  test_with_wrong_granularity<FloorTestGraph>();
}

TEST(QuantizedModelVerifierTest, GreaterEqual)
{
  test_with_graph<ComparisonOpTestGraph<luci::CircleGreaterEqual>>();
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_type_NEG)
{
  test_with_wrong_type<ComparisonOpTestGraph<luci::CircleGreaterEqual>>();
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_granularity_NEG)
{
  // TODO: Add x() getter function to the ComparisonOpTestGraph
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g._op->x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g._y);
}

TEST(QuantizedModelVerifierTest, Greater)
{
  test_with_graph<ComparisonOpTestGraph<luci::CircleGreater>>();
}

TEST(QuantizedModelVerifierTest, Greater_wrong_type_NEG)
{
  test_with_wrong_type<ComparisonOpTestGraph<luci::CircleGreater>>();
}

TEST(QuantizedModelVerifierTest, Greater_wrong_granularity_NEG)
{
  // TODO: Add x() getter function to the ComparisonOpTestGraph
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g._op->x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g._y);
}

TEST(QuantizedModelVerifierTest, NotEqual)
{
  test_with_graph<ComparisonOpTestGraph<luci::CircleNotEqual>>();
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_type_NEG)
{
  test_with_wrong_type<ComparisonOpTestGraph<luci::CircleNotEqual>>();
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_granularity_NEG)
{
  // TODO: Add x() getter function to the ComparisonOpTestGraph
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g._op->x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g._op->x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g._y);
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g._y);
}

TEST(QuantizedModelVerifierTest, Div) { test_with_graph<DivTestGraph>(); }

TEST(QuantizedModelVerifierTest, Div_wrong_type_NEG) { test_with_wrong_type<DivTestGraph>(); }

TEST(QuantizedModelVerifierTest, Div_wrong_granularity_NEG)
{
  test_with_wrong_granularity_with_target_x<DivTestGraph>();
  test_with_wrong_granularity_with_target_y<DivTestGraph>();
}

TEST(QuantizedModelVerifierTest, FloorDiv) { test_with_graph<FloorDivTestGraph>(); }

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_type_NEG)
{
  test_with_wrong_type<FloorDivTestGraph>();
}

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_granularity_NEG)
{
  test_with_wrong_granularity_with_target_x<FloorDivTestGraph>();
  test_with_wrong_granularity_with_target_y<FloorDivTestGraph>();
}

#undef TEST_WITH_GRAPH
#undef TEST_WITH_WRONG_TYPE
#undef TEST_WITH_WRONG_GRANULARITY
