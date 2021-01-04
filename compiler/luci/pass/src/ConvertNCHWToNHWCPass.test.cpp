/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <logo/Phase.h>

#include "luci/Pass/ConvertNCHWToNHWCPass.h"
#include "luci/Pass/ShapeInferencePass.h"

// TODO: Remove this after refactoring is done
#include "luci/Pass/MigrateLegacyShapeDtypePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph with a single Op (example: Add).
 *
 *  BEFORE
 *  - All Ops including Input/Output are NCHW.
 *
 *             [Input] [beta]
 *                |  /
 *              [Add]
 *                |
 *             [Output]
 *
 *  AFTER
 *  - All Ops including Input/Output are NHWC.
 *
 *             [Input]
 *                |
 *         [Transpose]
 *                |
 *        [Transpose] [beta]
 *                |  /
 *              [Add]
 *                |
 *         [Transpose]
 *                |
 *         [Transpose]
 *                |
 *             [Output]
 */
class SimpleGraph
{
public:
  SimpleGraph() = default;

public:
  void init()
  {
    input = g.nodes()->create<luci::CircleInput>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(loco::DataType::FLOAT32);
    input->dtype(loco::DataType::FLOAT32);
    output->dtype(loco::DataType::FLOAT32);
    graph_output->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    graph_input->shape({1, channel_size, 4, 4});
    input->shape({1, channel_size, 4, 4});
    output->shape({1, channel_size, 4, 4});
    graph_output->shape({1, channel_size, 4, 4});

    auto graph_body = insertGraphBody(input);
    output->from(graph_body);
  }

  virtual ~SimpleGraph() = default;

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class AddGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();

    add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    add->shape({1, channel_size, 4, 4});
    beta->shape({1, channel_size, 1, 1});

    beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }

    add->x(input);
    add->y(beta);

    return add;
  }

public:
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
};

class MulGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mul = g.nodes()->create<luci::CircleMul>();
    gamma = g.nodes()->create<luci::CircleConst>();

    mul->dtype(loco::DataType::FLOAT32);
    gamma->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    mul->shape({1, channel_size, 4, 4});
    gamma->shape({1, channel_size, 1, 1});

    gamma->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      gamma->at<loco::DataType::FLOAT32>(i) = i;
    }

    mul->x(input);
    mul->y(gamma);

    return mul;
  }

public:
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *gamma = nullptr;
};

void check_pre_trans(loco::Node *node)
{
  auto pre_trans = dynamic_cast<luci::CircleTranspose *>(node);
  auto pre_trans_perm = dynamic_cast<luci::CircleConst *>(pre_trans->perm());
  EXPECT_NE(nullptr, pre_trans);
  EXPECT_NE(nullptr, pre_trans_perm);
  EXPECT_EQ(1, pre_trans_perm->rank());
  EXPECT_EQ(4, pre_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, pre_trans_perm->dtype());
  EXPECT_EQ(0, pre_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, pre_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, pre_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, pre_trans_perm->at<loco::DataType::S32>(3));
}

void check_post_trans(loco::Node *node)
{
  auto post_trans = dynamic_cast<luci::CircleTranspose *>(node);
  auto post_trans_perm = dynamic_cast<luci::CircleConst *>(post_trans->perm());
  EXPECT_NE(nullptr, post_trans);
  EXPECT_NE(nullptr, post_trans_perm);
  EXPECT_EQ(1, post_trans_perm->rank());
  EXPECT_EQ(4, post_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, post_trans_perm->dtype());
  EXPECT_EQ(0, post_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(3, post_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, post_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(2, post_trans_perm->at<loco::DataType::S32>(3));
}

void run_phase(loco::Graph *g)
{
  logo::Phase phase;

  // Default passes.
  // TODO: Remove this after refactoring is done
  phase.emplace_back(std::make_unique<luci::MigrateLegacyShapeDtypePass>());
  phase.emplace_back(std::make_unique<luci::ShapeInferencePass>());

  // Pass to test
  phase.emplace_back(std::make_unique<luci::ConvertNCHWToNHWCPass>());

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

} // namespace

TEST(ConvertNCHWToNHWC, Add)
{
  AddGraph g;
  g.init();

  run_phase(&g.g);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.add->x());

  auto add_succs = loco::succs(g.add);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(1, new_beta->dim(1).value());
  EXPECT_EQ(1, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Mul)
{
  MulGraph g;
  g.init();

  run_phase(&g.g);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.mul->x());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  uint32_t channel_size = 16;
  auto new_gamma = dynamic_cast<luci::CircleConst *>(g.mul->y());
  EXPECT_NE(nullptr, new_gamma);
  EXPECT_EQ(4, new_gamma->rank());
  EXPECT_EQ(1, new_gamma->dim(0).value());
  EXPECT_EQ(1, new_gamma->dim(1).value());
  EXPECT_EQ(1, new_gamma->dim(2).value());
  EXPECT_EQ(channel_size, new_gamma->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Unknown_Shape_NEG)
{
  AddGraph g;
  g.init();

  // Unknown shape
  g.input->dim(0).unset();
  g.add->dim(0).unset();
  g.output->dim(0).unset();

  luci::ConvertNCHWToNHWCPass pass;
  EXPECT_EQ(false, pass.run(&g.g));
}
