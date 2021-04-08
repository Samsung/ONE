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

#include "luci/Pass/ResolveCustomOpBatchMatMulPass.h"

#include <luci/IR/CircleNodes.h>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

const int N = 1;
const int C = 2;
const int H_X = 1;
const int W_X = 4;
const int H_Y = 4;
const int W_Y = 4;

/**
 *  graph having Custom operator BatchMatMulV2
 *
 *  [CircleInput]  [CircleInput]
 *         \         /
 *       [CircleCustom]
 *             |
 *      [CircleCustomOut]
 *             |
 *       [CircleOutput]
 */
class BatchMatmulV2Graphlet
{
public:
  BatchMatmulV2Graphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // custom option
    auto flatbuffer_builder =
      std::unique_ptr<flatbuffers::FlatBufferBuilder>(new flatbuffers::FlatBufferBuilder(1024));
    auto flex_buffers = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_buffers->StartMap();
    flex_buffers->Bool("adj_x", false);
    flex_buffers->Bool("adj_y", false);
    flex_buffers->Int("T", 0 /* circle::TensorType_FLOAT32 */);
    flex_buffers->EndMap(map_start);
    flex_buffers->Finish();

    // CircleCustom(BatchMatMulV2, adj_x=False, adj_y=False)
    _batchmatmulv2 = g->nodes()->create<luci::CircleCustom>(2, 1);
    _batchmatmulv2->custom_code("BatchMatMulV2");
    _batchmatmulv2->custom_options(flex_buffers->GetBuffer());
    _batchmatmulv2->shape({N, C, H_X, W_Y});
    _batchmatmulv2->dtype(loco::DataType::FLOAT32);
    _batchmatmulv2->name("batchmatmulv2");

    // CircleCustomOut
    _batchmatmulv2_out = g->nodes()->create<luci::CircleCustomOut>();
    _batchmatmulv2_out->shape({N, C, H_X, W_Y});
    _batchmatmulv2_out->dtype(loco::DataType::FLOAT32);
    _batchmatmulv2_out->index(0);
  }

public:
  luci::CircleCustom *batchmatmulv2() { return _batchmatmulv2; }

protected:
  luci::CircleCustom *_batchmatmulv2 = nullptr;
  luci::CircleCustomOut *_batchmatmulv2_out = nullptr;
};

class BatchMatmulV2Graph : public TestIsGraphlet<2>,
                           public TestOGraphlet,
                           public BatchMatmulV2Graphlet
{
public:
  BatchMatmulV2Graph() = default;

  void init(void)
  {
    TestIsGraphlet<2>::init(g(), {{N, C, H_X, W_X}, {N, C, H_X, W_X}});
    TestOGraphlet::init(g(), {{N, C, H_X, W_Y}});
    BatchMatmulV2Graphlet::init(g());

    // TODO how set multiple of shape vector for TestIsGraphlet?
    // update shape for second input
    input(1)->shape({N, C, H_Y, W_Y});

    // connect graph
    _batchmatmulv2->inputs(0, input(0));
    _batchmatmulv2->inputs(1, input(1));
    _batchmatmulv2_out->input(_batchmatmulv2);

    output()->from(_batchmatmulv2_out);
  }
};

class BatchMatmulV2GraphTest : public ::testing::Test
{
public:
  BatchMatmulV2Graph g;
  luci::ResolveCustomOpBatchMatMulPass pass;
};

} // namespace

TEST(ResolveCustomOpBatchMatMulPassTest, name)
{
  luci::ResolveCustomOpBatchMatMulPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

/**
 *  Optimized graph looks like below.
 *
 *  [CircleInput]
 *        |
 *  [CircleBatchMatMul]
 *        |
 *  [CircleOutput]
 */
TEST_F(BatchMatmulV2GraphTest, simple_test)
{
  g.init();

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto batchmatmul = dynamic_cast<luci::CircleBatchMatMul *>(g.output()->from());
  EXPECT_NE(nullptr, batchmatmul);

  auto input_0 = dynamic_cast<luci::CircleInput *>(batchmatmul->x());
  auto input_1 = dynamic_cast<luci::CircleInput *>(batchmatmul->y());
  EXPECT_NE(nullptr, input_0);
  EXPECT_NE(nullptr, input_1);
}

TEST_F(BatchMatmulV2GraphTest, wrong_condition_NEG)
{
  g.init();

  // wrong custom code
  g.batchmatmulv2()->custom_code("BatchMatMulv2"); // v is lower case
  auto ret = pass.run(g.g());

  EXPECT_EQ(false, ret);
}
