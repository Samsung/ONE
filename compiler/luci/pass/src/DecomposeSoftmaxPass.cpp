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

#include "luci/Pass/DecomposeSoftmaxPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  BEFORE
 *        [CircleNode]
 *              |
 *              |
 *      [CircleSoftmax]
 *              |
 *              |
 *        [CircleNode]
 *
 *
 *  AFTER
 *
 *      [CircleNode]   [CircleConst(=-1)]
 *          |    \       /           |
 *          |     \     /            |
 *          | [CircleReduceMax]      |
 *          |    /                   |
 *          |   /                    |
 *          |  /                     |
 *        [Sub]                      |
 *          |                        |
 *          | [CircleConst(=beta)]   |
 *          |   /                    |
 *          |  /                     |
 *        [Mul] (if beta != 1)       |
 *          |                        |
 *        [Exp]                      |
 *          | \                      |
 *          |  \                     |
 *          |  [CircleSum]-----------+
 *          |  /
 *          | /
 *        [Div]
 *          |
 *          |
 *      [CircleNode]
 */
bool decompose_softmax(luci::CircleSoftmax *softmax)
{
  if (!softmax)
    return false;

  if (softmax->dtype() != loco::DataType::FLOAT32)
    return false;

  auto const input = loco::must_cast<luci::CircleNode *>(softmax->logits());
  auto g = softmax->graph();

  auto const beta = softmax->beta();
  auto const name = softmax->name();
  assert(name.length() > 0);

  // fill reduction index (-1) for CircleReduceMax and CircleSum
  auto index_const = g->nodes()->create<luci::CircleConst>();
  index_const->shape({}); // scalar
  index_const->dtype(loco::DataType::S32);
  index_const->rank(0);
  index_const->size<loco::DataType::S32>(1);
  index_const->at<loco::DataType::S32>(0) = -1;
  index_const->name(name + "/Softmax/reduction_index");
  luci::add_origin(index_const, luci::get_origin(softmax));

  // Create CircleReduceMax operation
  auto max = g->nodes()->create<luci::CircleReduceMax>();
  max->input(input);
  max->reduction_indices(index_const);
  max->keep_dims(true);
  max->name(name + "/Softmax/max");
  luci::add_origin(max, luci::get_origin(softmax));

  // Create CircleSub operation
  auto sub = g->nodes()->create<luci::CircleSub>();
  sub->x(input);
  sub->y(max);
  sub->fusedActivationFunction(luci::FusedActFunc::NONE);
  sub->name(name + "/Softmax/sub");
  luci::add_origin(sub, luci::get_origin(softmax));

  // input for exp can be either sub or mul (in case beta != 1)
  loco::Node *exp_input = sub;

  // multiply sub by beta in case it is nonunit
  if (std::abs(beta - 1.f) > 1.e-05f)
  {
    // Create constant for beta
    auto beta_const = g->nodes()->create<luci::CircleConst>();
    beta_const->shape({}); // scalar
    beta_const->dtype(loco::DataType::FLOAT32);
    beta_const->rank(0);
    beta_const->size<loco::DataType::FLOAT32>(1);
    beta_const->at<loco::DataType::FLOAT32>(0) = beta;
    beta_const->name(name + "/Softmax/beta_const");
    luci::add_origin(beta_const, luci::get_origin(softmax));

    // Create CircleMul
    auto mul = g->nodes()->create<luci::CircleMul>();
    mul->x(sub);
    mul->y(beta_const);
    mul->fusedActivationFunction(luci::FusedActFunc::NONE);
    mul->name(name + "/Softmax/beta_mul");
    luci::add_origin(mul, luci::get_origin(softmax));

    exp_input = mul;
  }

  // Create CircleExp operation
  auto exp = g->nodes()->create<luci::CircleExp>();
  exp->x(exp_input);
  exp->name(name + "/Softmax/exp");
  luci::add_origin(exp, luci::get_origin(softmax));

  // Create CircleSum operation
  auto sum = g->nodes()->create<luci::CircleSum>();
  sum->input(exp);
  sum->reduction_indices(index_const);
  sum->keep_dims(true);
  sum->name(name + "/Softmax/sum");
  luci::add_origin(sum, luci::get_origin(softmax));

  // Create CircleDiv operation
  auto div = g->nodes()->create<luci::CircleDiv>();
  div->x(exp);
  div->y(sum);
  div->fusedActivationFunction(luci::FusedActFunc::NONE);
  div->name(name + "/Softmax/div");
  luci::add_origin(div, luci::get_origin(softmax));

  replace(softmax).with(div);

  return true;
}

} // namespace

namespace luci
{

bool DecomposeSoftmaxPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto softmax = dynamic_cast<luci::CircleSoftmax *>(node))
    {
      if (decompose_softmax(softmax))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
