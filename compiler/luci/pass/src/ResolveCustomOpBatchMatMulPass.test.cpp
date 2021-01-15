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

#include <gtest/gtest.h>

namespace
{

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
class BatchMatmulV2Graph : public ::testing::Test
{
protected:
  BatchMatmulV2Graph()
  {
    const int N = 1;
    const int C = 2;
    const int H_X = 1;
    const int W_X = 4;
    const int H_Y = 4;
    const int W_Y = 4;

    // graph input and output
    auto graph_input_0 = _g.inputs()->create();
    auto graph_input_1 = _g.inputs()->create();
    auto graph_output = _g.outputs()->create();

    // CircleInput
    _input_0 = _g.nodes()->create<luci::CircleInput>();
    _input_0->index(graph_input_0->index());
    _input_0->shape({N, C, H_X, W_X});
    _input_0->dtype(loco::DataType::FLOAT32);
    _input_1 = _g.nodes()->create<luci::CircleInput>();
    _input_1->index(graph_input_1->index());
    _input_1->shape({N, C, H_Y, W_Y});
    _input_1->dtype(loco::DataType::FLOAT32);

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
    _batchmatmulv2 = _g.nodes()->create<luci::CircleCustom>(2);
    _batchmatmulv2->custom_code("BatchMatMulV2");
    _batchmatmulv2->custom_options(flex_buffers->GetBuffer());
    _batchmatmulv2->inputs(0, _input_0);
    _batchmatmulv2->inputs(1, _input_1);
    _batchmatmulv2->shape({N, C, H_X, W_Y});
    _batchmatmulv2->dtype(loco::DataType::FLOAT32);

    // CircleCustomOut
    _batchmatmulv2_out = _g.nodes()->create<luci::CircleCustomOut>();
    _batchmatmulv2_out->input(_batchmatmulv2);
    _batchmatmulv2_out->index(0);
    _batchmatmulv2_out->shape({N, C, H_X, W_Y});
    _batchmatmulv2_out->dtype(loco::DataType::FLOAT32);

    // CircleOutput
    _output = _g.nodes()->create<luci::CircleOutput>();
    _output->index(graph_output->index());
    _output->from(_batchmatmulv2_out);
    _output->shape({N, C, H_X, W_Y});
    _output->dtype(loco::DataType::FLOAT32);
  }

protected:
  loco::Graph _g;
  luci::CircleInput *_input_0 = nullptr;
  luci::CircleInput *_input_1 = nullptr;
  luci::CircleCustom *_batchmatmulv2 = nullptr;
  luci::CircleCustomOut *_batchmatmulv2_out = nullptr;
  luci::CircleOutput *_output = nullptr;
};

} // namespace

/**
 *  Optimized graph looks like below.
 *
 *  [CircleInput]
 *        |
 *  [CircleBatchMatMul]
 *        |
 *  [CircleOutput]
 */
TEST_F(BatchMatmulV2Graph, simple_test)
{
  luci::ResolveCustomOpBatchMatMulPass pass;
  auto ret = pass.run(&_g);
  EXPECT_EQ(true, ret);

  auto batchmatmul = dynamic_cast<luci::CircleBatchMatMul *>(_output->from());
  EXPECT_NE(nullptr, batchmatmul);

  auto input_0 = dynamic_cast<luci::CircleInput *>(batchmatmul->x());
  auto input_1 = dynamic_cast<luci::CircleInput *>(batchmatmul->y());
  EXPECT_NE(nullptr, input_0);
  EXPECT_NE(nullptr, input_1);
}

TEST_F(BatchMatmulV2Graph, wrong_condition_NEG)
{
  luci::ResolveCustomOpBatchMatMulPass pass;

  // wrong custom code
  _batchmatmulv2->custom_code("BatchMatMulv2");
  auto ret = pass.run(&_g);

  EXPECT_EQ(false, ret);
}
