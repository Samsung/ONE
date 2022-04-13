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

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Pass/ReplaceNonConstFCWithBatchMatMulPass.h>

namespace
{

/**
 *  Replace channel-wise Mul/Add with DepthwiseConv2D
 *
 *  BEFORE
 *
 *         [Node1]         [Node2]
 *           |               |
 *       [transpose]?   [transpose]?
 *               \        /
 *           [ Fully Connected ]
 *
 *  AFTER
 *
 *              [Node1]  [Node2]
 *                  \      /
 *                [BatchMatMul]
 *
 */
bool replace_fc_with_matmul(luci::CircleFullyConnected *fc)
{
  luci::CircleNode *x = nullptr;
  luci::CircleNode *y = nullptr;
  luci::CircleNode *b = nullptr;
  luci::CircleTranspose *ty = nullptr;
  luci::CircleTranspose *tx = nullptr;
  bool adj_x = false;
  bool adj_y = true;

  if (dynamic_cast<luci::CircleConst *>(fc->weights()))
    return false; // NonConst

  if ((ty = dynamic_cast<luci::CircleTranspose *>(fc->weights()))) // is y a transpose?
  {
    adj_y = false;
    if (dynamic_cast<luci::CircleConst *>(ty->a()))
      return false;
    else
      y = loco::must_cast<luci::CircleNode *>(ty->a());
  }
  else
  { // y is not transpose and not const
    y = loco::must_cast<luci::CircleNode *>(fc->weights());
  }
  if ((tx = dynamic_cast<luci::CircleTranspose *>(fc->input())))
  {
    adj_x = true;
    x = loco::must_cast<luci::CircleNode *>(tx->a());
  }
  else
  {
    x = loco::must_cast<luci::CircleNode *>(fc->input());
  }

  b = loco::must_cast<luci::CircleNode *>(fc->bias());

  if (x->dtype() != loco::DataType::FLOAT32 || y->dtype() != loco::DataType::FLOAT32 ||
      b->dtype() != loco::DataType::FLOAT32)
    return false;

  auto name = fc->name();
  assert(name.length() > 0);

  auto matmul = fc->graph()->nodes()->create<luci::CircleBatchMatMul>();
  matmul->x(x);
  matmul->y(y);
  matmul->adj_x(adj_x);
  matmul->adj_y(adj_y);
  matmul->name(name);
  matmul->dtype(fc->dtype());

  luci::add_origin(matmul, luci::get_origin(fc));

  auto all_zero = [](const luci::CircleConst *c) {
    bool ac = true;
    for (uint32_t i = 0; i < c->size<loco::DataType::FLOAT32>() && ac; i++)
    {
      ac &= c->at<loco::DataType::FLOAT32>(i) == 0.0f;
    }
    return ac;
  };

  luci::CircleConst *bc;
  if ((bc = dynamic_cast<luci::CircleConst *>(b)) && !all_zero(bc))
  {
    auto bias = fc->graph()->nodes()->create<luci::CircleAdd>();
    bias->x(matmul);
    bias->y(b);
    bias->name(fc->name() + "/bias");
    bias->dtype(fc->dtype());
    loco::replace(fc).with(bias);
  }
  else
  {
    loco::replace(fc).with(matmul);
  }

  return true;
}

} // namespace

namespace luci
{

bool ReplaceNonConstFCWithBatchMatMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto fc = dynamic_cast<luci::CircleFullyConnected *>(node))
    {
      if (replace_fc_with_matmul(fc))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
