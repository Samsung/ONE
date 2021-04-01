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

#include "test/TestIOGraph.h"

#include <gtest/gtest.h>

using Type = loco::DataType;
using Granularity = luci::QuantizationGranularity;

namespace
{

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

// Set min/max for all non-const nodes in the graph
void set_minmax_to_non_const(loco::Graph *g, float min, float max)
{
  for (auto node : loco::all_nodes(g))
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (const_node != nullptr)
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

} // namespace

// Quantize and verify with given configurations
#define TEST_WITH_GRAPH(graph, type, granularity)                   \
  do                                                                \
  {                                                                 \
    graph g;                                                        \
    g.init();                                                       \
    EXPECT_NO_THROW(quantize_and_verify(g.g(), type, granularity)); \
  } while (0)

TEST(QuantizedModelVerifierTest, Logistic)
{
  {
    LogisticTestGraph g;
    g.init();
    EXPECT_NO_THROW(quantize_and_verify(g.g(), Type::U8, Granularity::LayerWise));
  }

  {
    LogisticTestGraph g;
    g.init();
    EXPECT_NO_THROW(quantize_and_verify(g.g(), Type::U8, Granularity::ChannelWise));
  }

  {
    LogisticTestGraph g;
    g.init();
    EXPECT_NO_THROW(quantize_and_verify(g.g(), Type::S16, Granularity::ChannelWise));
  }
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_type_NEG)
{
  {
    LogisticTestGraph g;
    g.init();

    luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
    pass.run(g.g());

    g._logistic->dtype(Type::S16);

    luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
    EXPECT_ANY_THROW(verifier.verify(g.g()));
  }
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_granularity_NEG)
{
  {
    LogisticTestGraph g;
    g.init();

    luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
    pass.run(g.g());

    insert_scale_zp(g._logistic, 1.0, 1);

    luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
    EXPECT_ANY_THROW(verifier.verify(g.g()));
  }
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
  {
    SliceTestGraph<Type::S32> g;
    g.init();

    luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
    pass.run(g.g());

    g._slice->dtype(Type::S16);

    luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
    EXPECT_ANY_THROW(verifier.verify(g.g()));
  }
}

TEST(QuantizedModelVerifierTest, Slice_wrong_granularity_NEG)
{
  {
    SliceTestGraph<Type::S32> g;
    g.init();

    luci::QuantizeWithMinMaxPass pass(Type::FLOAT32, Type::U8, Granularity::LayerWise);
    pass.run(g.g());

    insert_scale_zp(g._slice, 1.0, 1);

    luci::QuantizedModelVerifier verifier(Type::U8, Granularity::LayerWise);
    EXPECT_ANY_THROW(verifier.verify(g.g()));
  }
}

#undef TEST_WITH_GRAPH
