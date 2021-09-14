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

#include "luci/Pass/ForceQuantParamPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

std::unique_ptr<luci::CircleQuantParam> make_qparam(float scale, int64_t zp)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  qparam->scale.push_back(scale);
  qparam->zerop.push_back(zp);

  return std::move(qparam);
}

bool check_per_tensor_qparam(luci::CircleNode *node, float scale, int64_t zp)
{
  assert(node); // FIX_CALLER_UNLESS

  auto qparam = node->quantparam();
  if (qparam->scale.size() != 1)
    return false;

  if (qparam->scale[0] != scale)
    return false;

  if (qparam->zerop.size() != 1)
    return false;

  if (qparam->zerop[0] != zp)
    return false;

  return true;
}

/**
 *  Graph with a single input and a single output.
 *
 *             [Input]
 *                |
 *           (graph body) -> implemented by insertGraphBody()
 *                |
 *             [Output]
 *
 */
class SISOGraph
{
public:
  SISOGraph() = default;

public:
  void init()
  {
    input = g.nodes()->create<luci::CircleInput>();
    output = g.nodes()->create<luci::CircleOutput>();
    input->name("input");
    output->name("output");

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(loco::DataType::U8);
    input->dtype(loco::DataType::U8);
    output->dtype(loco::DataType::U8);
    graph_output->dtype(loco::DataType::U8);

    input->quantparam(make_qparam(0.1, 11));
    output->quantparam(make_qparam(0.2, 12));

    uint32_t channel_size = 16;
    graph_input->shape({1, channel_size, 4, 4});
    input->shape({1, channel_size, 4, 4});
    output->shape({1, channel_size, 4, 4});
    graph_output->shape({1, channel_size, 4, 4});

    auto graph_body = insertGraphBody(input);
    output->from(graph_body);
  }

  virtual ~SISOGraph() = default;

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class AddGraph final : public SISOGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();

    add->dtype(loco::DataType::U8);
    beta->dtype(loco::DataType::U8);
    add->quantparam(make_qparam(0.1, 11));
    beta->quantparam(make_qparam(0.2, 12));

    uint32_t channel_size = 16;
    add->shape({1, 4, 4, channel_size});
    beta->shape({1, 1, 1, channel_size});

    beta->size<loco::DataType::U8>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::U8>(i) = i;
    }

    add->x(input);
    add->y(beta);

    add->name("add");
    beta->name("beta");

    return add;
  }

public:
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
};

} // namespace

TEST(ForceQuantParamPassTest, simple)
{
  TensorVector tensors{"input", "add"};
  ScaleVector scales{2.0, 3.0};
  ZPVector zerops{4, 8};

  luci::ForceQuantParamPass pass(tensors, scales, zerops);

  AddGraph g;
  g.init();

  pass.run(&g.g);

  EXPECT_TRUE(check_per_tensor_qparam(g.input, 2.0, 4));
  EXPECT_TRUE(check_per_tensor_qparam(g.add, 3.0, 8));
}

TEST(ForceQuantParamPassTest, name_mismatch_NEG)
{
  TensorVector tensors{"no_exist"};
  ScaleVector scales{2.0};
  ZPVector zerops{4};

  luci::ForceQuantParamPass pass(tensors, scales, zerops);

  AddGraph g;
  g.init();

  EXPECT_THROW(pass.run(&g.g), std::runtime_error);
}
