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

#include "luci/Pass/UnrollUnidirectionalSequenceLSTMPass.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/Nodes/CircleUnidirectionalSequenceLSTM.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class UniSeqLSTMGraphlet
{
public:
  UniSeqLSTMGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 oshape)
  {
    _uslstm = g->nodes()->create<luci::CircleUnidirectionalSequenceLSTM>();
    _uslstm->dtype(loco::DataType::FLOAT32);
    _uslstm->shape(oshape);
    _uslstm->name("uslstm");

    _uslstm->fusedActivationFunction(luci::FusedActFunc::TANH);
    _uslstm->cell_clip(0.0);
    _uslstm->proj_clip(0.0);
    _uslstm->time_major(false);
    _uslstm->asymmetric_quantize_inputs(false);

    _iw = weight_1x1(g);
    _rw = weight_1x1(g);
    _gb = weight_1(g);
    _ex = g->nodes()->create<luci::CircleOutputExclude>();
  }

protected:
  luci::CircleConst *weight_1x1(loco::Graph *g)
  {
    auto w = g->nodes()->create<luci::CircleConst>();
    w->dtype(loco::DataType::FLOAT32);
    w->rank(2);
    w->dim(0) = 1;
    w->dim(1) = 1;
    w->size<loco::DataType::FLOAT32>(1);
    w->at<loco::DataType::FLOAT32>(0) = 1.0;
    w->shape_status(luci::ShapeStatus::VALID);
    return w;
  }

  luci::CircleConst *weight_1(loco::Graph *g)
  {
    auto w = g->nodes()->create<luci::CircleConst>();
    w->dtype(loco::DataType::FLOAT32);
    w->rank(1);
    w->dim(0) = 1;
    w->size<loco::DataType::FLOAT32>(1);
    w->at<loco::DataType::FLOAT32>(0) = 1.0;
    w->shape_status(luci::ShapeStatus::VALID);
    return w;
  }

protected:
  luci::CircleUnidirectionalSequenceLSTM *_uslstm = nullptr;
  luci::CircleConst *_iw = nullptr;
  luci::CircleConst *_rw = nullptr;
  luci::CircleConst *_gb = nullptr;
  luci::CircleOutputExclude *_ex = nullptr;
};

class UnrollUniSeqLSTMPassTestGraph : public TestIOGraph, public UniSeqLSTMGraphlet
{
public:
  UnrollUniSeqLSTMPassTestGraph() = default;

  void init(const ShapeU32 ishape, const ShapeU32 oshape)
  {
    TestIOGraph::init(ishape, oshape);
    UniSeqLSTMGraphlet::init(g(), oshape);

    auto inode = input();
    _uslstm->input(inode);

    _uslstm->input_to_input_weights(_iw);
    _uslstm->input_to_forget_weights(_iw);
    _uslstm->input_to_cell_weights(_iw);
    _uslstm->input_to_output_weights(_iw);

    _uslstm->recurrent_to_input_weights(_rw);
    _uslstm->recurrent_to_forget_weights(_rw);
    _uslstm->recurrent_to_cell_weights(_rw);
    _uslstm->recurrent_to_output_weights(_rw);

    _uslstm->cell_to_input_weights(_ex);
    _uslstm->cell_to_forget_weights(_ex);
    _uslstm->cell_to_output_weights(_ex);

    _uslstm->input_gate_bias(_gb);
    _uslstm->forget_gate_bias(_gb);
    _uslstm->cell_gate_bias(_gb);
    _uslstm->output_gate_bias(_gb);

    _uslstm->projection_weights(_ex);
    _uslstm->projection_bias(_ex);

    _uslstm->output_state(_ex);
    _uslstm->cell_state(_ex);

    _uslstm->input_layer_norm_coefficients(_ex);
    _uslstm->forget_layer_norm_coefficients(_ex);
    _uslstm->cell_layer_norm_coefficients(_ex);
    _uslstm->output_layer_norm_coefficients(_ex);

    output()->from(_uslstm);
  }
};

} // namespace

namespace
{

using namespace luci::test;

// FakeQuantGraphlet is for simple negative test
class FakeQuantGraphlet
{
public:
  FakeQuantGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _fq = g->nodes()->create<luci::CircleFakeQuant>();
    _fq->name("fq");
  }

protected:
  luci::CircleFakeQuant *_fq = nullptr;
};

class FakeQuantGraph : public TestIOGraph, public FakeQuantGraphlet
{
public:
  FakeQuantGraph() = default;

public:
  void init(void)
  {
    TestIOGraph::init({1, 1, 1}, {1, 1, 1});
    FakeQuantGraphlet::init(g());

    _fq->inputs(input());

    output()->from(_fq);
  }
};

} // namespace

TEST(UnrollUnidirectionalSequenceLSTMPassTestName, name)
{
  luci::UnrollUnidirectionalSequenceLSTMPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

class UnrollUnidirectionalSequenceLSTMPassTest : public ::testing::Test
{
public:
  UnrollUniSeqLSTMPassTestGraph g;
  luci::UnrollUnidirectionalSequenceLSTMPass pass;
};

TEST_F(UnrollUnidirectionalSequenceLSTMPassTest, simple_run)
{
  g.init({1, 1, 1}, {1, 1, 1});

  EXPECT_TRUE(pass.run(g.g()));
}

class UnrollUnidirectionalSequenceLSTMPassTestN : public ::testing::Test
{
public:
  FakeQuantGraph g;
  luci::UnrollUnidirectionalSequenceLSTMPass pass;
};

TEST_F(UnrollUnidirectionalSequenceLSTMPassTestN, simple_run_NEG)
{
  g.init();

  EXPECT_FALSE(pass.run(g.g()));
}
