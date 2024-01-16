/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseGRUPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class GRUGraphlet
{
public:
  GRUGraphlet() = default;

  void init(loco::Graph *g)
  {
    _while_node = g->nodes()->create<luci::CircleWhile>(5, 5);
    _while_out_node = g->nodes()->create<luci::CircleWhileOut>();
    _hidden_node = g->nodes()->create<luci::CircleConst>();
    _hidden_node->dtype(loco::DataType::FLOAT32);
    _time_node = g->nodes()->create<luci::CircleConst>();
    _time_node->dtype(loco::DataType::FLOAT32);
    _state_node = g->nodes()->create<luci::CircleConst>();
    _state_node->dtype(loco::DataType::FLOAT32);

    _body_graph = loco::make_graph();
    _cond_graph = loco::make_graph();

    _less_node = _cond_graph->nodes()->create<luci::CircleLess>();
    _less_const_node = _cond_graph->nodes()->create<luci::CircleConst>();
    _less_const_node->dtype(loco::DataType::S32);
    _less_const_node->size<loco::DataType::S32>(1);
    _less_const_node->at<loco::DataType::S32>(0) = 1;

    _add_node_1 = _body_graph->nodes()->create<luci::CircleAdd>();
    _add_node_2 = _body_graph->nodes()->create<luci::CircleAdd>();
    _add_node_3 = _body_graph->nodes()->create<luci::CircleAdd>();
    _add_node_4 = _body_graph->nodes()->create<luci::CircleAdd>();
    _add_node_5 = _body_graph->nodes()->create<luci::CircleAdd>();
    _add_node_6 = _body_graph->nodes()->create<luci::CircleAdd>();

    _fc_node_1 = _body_graph->nodes()->create<luci::CircleFullyConnected>();
    _fc_node_2 = _body_graph->nodes()->create<luci::CircleFullyConnected>();
    _fc_weight_1 = _body_graph->nodes()->create<luci::CircleConst>();
    _fc_weight_1->dtype(loco::DataType::FLOAT32);
    _fc_weight_2 = _body_graph->nodes()->create<luci::CircleConst>();
    _fc_weight_2->dtype(loco::DataType::FLOAT32);
    _fc_bias_1 = _body_graph->nodes()->create<luci::CircleConst>();
    _fc_bias_1->dtype(loco::DataType::FLOAT32);
    _fc_bias_2 = _body_graph->nodes()->create<luci::CircleConst>();
    _fc_bias_2->dtype(loco::DataType::FLOAT32);

    _logistic_node_1 = _body_graph->nodes()->create<luci::CircleLogistic>();
    _logistic_node_2 = _body_graph->nodes()->create<luci::CircleLogistic>();

    _gather_node = _body_graph->nodes()->create<luci::CircleGather>();

    _mul_node_1 = _body_graph->nodes()->create<luci::CircleMul>();
    _mul_node_2 = _body_graph->nodes()->create<luci::CircleMul>();
    _mul_node_3 = _body_graph->nodes()->create<luci::CircleMul>();

    _tanh_node = _body_graph->nodes()->create<luci::CircleTanh>();
    _sub_node = _body_graph->nodes()->create<luci::CircleSub>();

    _split_node_1 = _body_graph->nodes()->create<luci::CircleSplit>();
    _split_node_2 = _body_graph->nodes()->create<luci::CircleSplit>();
    _split_out_node_1 = _body_graph->nodes()->create<luci::CircleSplitOut>();
    _split_out_node_2 = _body_graph->nodes()->create<luci::CircleSplitOut>();
    _split_out_node_3 = _body_graph->nodes()->create<luci::CircleSplitOut>();
    _split_out_node_4 = _body_graph->nodes()->create<luci::CircleSplitOut>();
    _split_out_node_5 = _body_graph->nodes()->create<luci::CircleSplitOut>();
    _split_out_node_6 = _body_graph->nodes()->create<luci::CircleSplitOut>();

    _reshape_node = _body_graph->nodes()->create<luci::CircleReshape>();

    auto graph_input_cond_graph = _cond_graph->inputs()->create();
    _cond_input_node = _cond_graph->nodes()->create<luci::CircleInput>();
    _cond_input_node->index(graph_input_cond_graph->index());

    auto graph_output_cond_graph = _cond_graph->outputs()->create();
    _cond_output_node = _cond_graph->nodes()->create<luci::CircleOutput>();
    _cond_output_node->index(graph_output_cond_graph->index());

    auto graph_input_body_graph_1 = _body_graph->inputs()->create();
    _body_input_node_1 = _body_graph->nodes()->create<luci::CircleInput>();
    _body_input_node_1->index(graph_input_body_graph_1->index());

    auto graph_input_body_graph_2 = _body_graph->inputs()->create();
    _body_input_node_2 = _body_graph->nodes()->create<luci::CircleInput>();
    _body_input_node_2->index(graph_input_body_graph_2->index());

    auto graph_input_body_graph_3 = _body_graph->inputs()->create();
    _body_input_node_3 = _body_graph->nodes()->create<luci::CircleInput>();
    _body_input_node_3->index(graph_input_body_graph_3->index());

    auto graph_input_body_graph_4 = _body_graph->inputs()->create();
    _body_input_node_4 = _body_graph->nodes()->create<luci::CircleInput>();
    _body_input_node_4->index(graph_input_body_graph_4->index());

    auto graph_input_body_graph_5 = _body_graph->inputs()->create();
    _body_input_node_5 = _body_graph->nodes()->create<luci::CircleInput>();
    _body_input_node_5->index(graph_input_body_graph_5->index());

    auto graph_output_body_graph_1 = _body_graph->outputs()->create();
    _body_output_node_1 = _body_graph->nodes()->create<luci::CircleOutput>();
    _body_output_node_1->index(graph_output_body_graph_1->index());

    auto graph_output_body_graph_2 = _body_graph->outputs()->create();
    _body_output_node_2 = _body_graph->nodes()->create<luci::CircleOutput>();
    _body_output_node_2->index(graph_output_body_graph_2->index());

    auto graph_output_body_graph_3 = _body_graph->outputs()->create();
    _body_output_node_3 = _body_graph->nodes()->create<luci::CircleOutput>();
    _body_output_node_3->index(graph_output_body_graph_3->index());

    auto graph_output_body_graph_4 = _body_graph->outputs()->create();
    _body_output_node_4 = _body_graph->nodes()->create<luci::CircleOutput>();
    _body_output_node_4->index(graph_output_body_graph_4->index());

    auto graph_output_body_graph_5 = _body_graph->outputs()->create();
    _body_output_node_5 = _body_graph->nodes()->create<luci::CircleOutput>();
    _body_output_node_5->index(graph_output_body_graph_5->index());
  }

  void invalid_less_const_type() { _less_const_node->dtype(loco::DataType::S16); }

protected:
  luci::CircleWhile *_while_node;
  luci::CircleWhileOut *_while_out_node;
  luci::CircleConst *_time_node;
  luci::CircleConst *_state_node;
  luci::CircleConst *_hidden_node;

  luci::CircleInput *_cond_input_node;
  luci::CircleLess *_less_node;
  luci::CircleConst *_less_const_node;
  luci::CircleOutput *_cond_output_node;

  luci::CircleInput *_body_input_node_1;
  luci::CircleInput *_body_input_node_2;
  luci::CircleInput *_body_input_node_3;
  luci::CircleInput *_body_input_node_4;
  luci::CircleInput *_body_input_node_5;

  luci::CircleOutput *_body_output_node_1;
  luci::CircleOutput *_body_output_node_2;
  luci::CircleOutput *_body_output_node_3;
  luci::CircleOutput *_body_output_node_4;
  luci::CircleOutput *_body_output_node_5;

  luci::CircleAdd *_add_node_1;
  luci::CircleAdd *_add_node_2;
  luci::CircleAdd *_add_node_3;
  luci::CircleAdd *_add_node_4;
  luci::CircleAdd *_add_node_5;
  luci::CircleAdd *_add_node_6;

  luci::CircleMul *_mul_node_1;
  luci::CircleMul *_mul_node_2;
  luci::CircleMul *_mul_node_3;

  luci::CircleSub *_sub_node;
  luci::CircleTanh *_tanh_node;
  luci::CircleReshape *_reshape_node;
  luci::CircleGather *_gather_node;
  luci::CircleLogistic *_logistic_node_1;
  luci::CircleLogistic *_logistic_node_2;
  luci::CircleSplit *_split_node_1;
  luci::CircleSplit *_split_node_2;

  luci::CircleSplitOut *_split_out_node_1;
  luci::CircleSplitOut *_split_out_node_2;
  luci::CircleSplitOut *_split_out_node_3;
  luci::CircleSplitOut *_split_out_node_4;
  luci::CircleSplitOut *_split_out_node_5;
  luci::CircleSplitOut *_split_out_node_6;

  luci::CircleFullyConnected *_fc_node_1;
  luci::CircleFullyConnected *_fc_node_2;

  luci::CircleConst *_fc_weight_1;
  luci::CircleConst *_fc_bias_1;
  luci::CircleConst *_fc_weight_2;
  luci::CircleConst *_fc_bias_2;

  std::unique_ptr<loco::Graph> _cond_graph;
  std::unique_ptr<loco::Graph> _body_graph;
};

class FuseGRUTestGraph1 : public TestIOGraph, public GRUGraphlet
{
public:
  FuseGRUTestGraph1() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    GRUGraphlet::init(g());

    _while_node->input(0, _time_node);
    _while_node->input(1, _time_node);
    _while_node->input(2, _state_node);
    _while_node->input(3, _hidden_node);
    _while_node->input(4, input());

    _while_out_node->input(_while_node);
    output()->from(_while_out_node);

    _while_node->cond_graph(_cond_graph.get());
    _while_node->body_graph(_body_graph.get());

    // cond graph
    _less_node->x(_cond_input_node);
    _less_node->y(_less_const_node);
    _cond_output_node->from(_less_node);

    // body graph
    _add_node_1->x(_body_input_node_1);
    _add_node_2->x(_body_input_node_2);

    _body_output_node_5->from(_add_node_1);
    _body_output_node_4->from(_add_node_2);

    _gather_node->params(_body_input_node_2);
    _fc_node_1->input(_body_input_node_4);
    _fc_node_1->weights(_fc_weight_1);
    _fc_node_1->bias(_fc_bias_1);
    _fc_node_2->input(_gather_node);
    _fc_node_2->weights(_fc_weight_2);
    _fc_node_2->bias(_fc_bias_2);

    _split_node_1->input(_fc_node_1);
    _split_node_2->input(_fc_node_2);

    _split_out_node_1->input(_split_node_1);
    _split_out_node_2->input(_split_node_1);
    _split_out_node_3->input(_split_node_1);

    _split_out_node_4->input(_split_node_2);
    _split_out_node_5->input(_split_node_2);
    _split_out_node_6->input(_split_node_2);

    _add_node_3->x(_split_out_node_1);
    _add_node_3->y(_split_out_node_4);

    _add_node_4->x(_split_out_node_3);
    _add_node_4->y(_split_out_node_6);

    _logistic_node_1->x(_add_node_3);

    _mul_node_1->x(_body_input_node_4);
    _mul_node_1->y(_logistic_node_1);

    _sub_node->y(_logistic_node_1);

    _logistic_node_2->x(_add_node_4);

    _mul_node_2->x(_split_out_node_2);
    _mul_node_2->y(_logistic_node_2);

    _add_node_5->x(_split_out_node_5);
    _add_node_5->y(_mul_node_2);

    _tanh_node->x(_add_node_5);

    _mul_node_3->x(_sub_node);
    _mul_node_3->y(_tanh_node);

    _add_node_6->x(_mul_node_1);
    _add_node_6->y(_mul_node_3);

    _reshape_node->shape(_add_node_6);

    _body_output_node_3->from(_reshape_node);
  }
};

class FuseGRUTestNegGraph : public TestIOGraph, public GRUGraphlet
{
public:
  FuseGRUTestNegGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    GRUGraphlet::init(g());

    invalid_less_const_type();

    _while_node->input(0, _time_node);
    _while_node->input(1, _time_node);
    _while_node->input(2, _state_node);
    _while_node->input(3, _hidden_node);
    _while_node->input(4, input());

    _while_node->cond_graph(_cond_graph.get());
    _while_node->body_graph(_body_graph.get());

    _while_out_node->input(_while_node);
    output()->from(_while_out_node);

    // cond graph
    _less_node->x(_cond_input_node);
    _less_node->y(_less_const_node);
    _cond_output_node->from(_less_node);

    // body graph
    _add_node_1->x(_body_input_node_1);
    _add_node_2->x(_body_input_node_2);

    _body_output_node_5->from(_add_node_1);
    _body_output_node_4->from(_add_node_2);

    _gather_node->params(_body_input_node_2);
    _fc_node_1->input(_body_input_node_4);
    _fc_node_1->weights(_fc_weight_1);
    _fc_node_1->bias(_fc_bias_1);
    _fc_node_2->input(_gather_node);
    _fc_node_2->weights(_fc_weight_2);
    _fc_node_2->bias(_fc_bias_2);

    _split_node_1->input(_fc_node_1);
    _split_node_2->input(_fc_node_2);

    _split_out_node_1->input(_split_node_1);
    _split_out_node_2->input(_split_node_1);
    _split_out_node_3->input(_split_node_1);

    _split_out_node_4->input(_split_node_2);
    _split_out_node_5->input(_split_node_2);
    _split_out_node_6->input(_split_node_2);

    _add_node_3->x(_split_out_node_1);
    _add_node_3->y(_split_out_node_4);

    _add_node_4->x(_split_out_node_3);
    _add_node_4->y(_split_out_node_6);

    _logistic_node_1->x(_add_node_3);

    _mul_node_1->x(_body_input_node_4);
    _mul_node_1->y(_logistic_node_1);

    _sub_node->y(_logistic_node_1);

    _logistic_node_2->x(_add_node_4);

    _mul_node_2->x(_split_out_node_2);
    _mul_node_2->y(_logistic_node_2);

    _add_node_5->x(_split_out_node_5);
    _add_node_5->y(_mul_node_2);

    _tanh_node->x(_add_node_5);

    _mul_node_3->x(_sub_node);
    _mul_node_3->y(_tanh_node);

    _add_node_6->x(_mul_node_1);
    _add_node_6->y(_mul_node_3);

    _reshape_node->shape(_add_node_6);

    _body_output_node_3->from(_reshape_node);
  }
};

} // namespace

TEST(FuseGRUPassTest, fuse_pattern1)
{
  FuseGRUTestGraph1 g;
  luci::FuseGRUPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseGRUPassTest, fuse_NEG)
{
  FuseGRUTestNegGraph g;
  luci::FuseGRUPass pass;

  g.init();

  EXPECT_FALSE(pass.run(g.g()));
}
