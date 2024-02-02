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

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Pass/ReplaceNonConstFCWithBatchMatMulPass.h>

namespace
{

// TODO move to global helper list if needed
/**
 * @brief Create a node with `inp` as input from fused activation fucntion `act`
 */
luci::CircleNode *fromActivation(luci::CircleNode *inp, luci::FusedActFunc act)
{
  switch (act)
  {
    case luci::FusedActFunc::NONE:
      return inp;
    case luci::FusedActFunc::RELU:
    {
      auto n = inp->graph()->nodes()->create<luci::CircleRelu>();
      n->features(inp);
      return n;
    }
    case luci::FusedActFunc::RELU6:
    {
      auto n = inp->graph()->nodes()->create<luci::CircleRelu6>();
      n->features(inp);
      return n;
    }
    case luci::FusedActFunc::RELU_N1_TO_1:
    {
      auto n = inp->graph()->nodes()->create<luci::CircleReluN1To1>();
      n->features(inp);
      return n;
    }
    case luci::FusedActFunc::TANH:
    {
      auto n = inp->graph()->nodes()->create<luci::CircleTanh>();
      n->x(inp);
      return n;
    }
    case luci::FusedActFunc::SIGN_BIT:
    {
      throw std::invalid_argument("no matching node to create from fused activation");
    }
    default:
      throw std::invalid_argument("invalid fused activation");
  }
}

// Create CircleReshape where
// - dtype is same with node
// - shape is same with node
// NOTE: User should set input(tensor) of the returned Op.
luci::CircleReshape *create_reshape(luci::CircleFullyConnected *node)
{
  assert(node); // FIX_CALLER_UNLESS

  auto g = node->graph();

  auto reshape = g->nodes()->create<luci::CircleReshape>();
  reshape->name(node->name() + "/reshape");
  reshape->dtype(node->dtype());
  luci::add_origin(reshape, luci::get_origin(node));

  auto shape_const = g->nodes()->create<luci::CircleConst>();
  shape_const->dtype(loco::DataType::S32);
  shape_const->rank(1);
  shape_const->dim(0).set(node->rank());
  shape_const->size<loco::DataType::S32>(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
  {
    assert(node->dim(i).known()); // FIX_CALLER_UNLESS
    shape_const->at<loco::DataType::S32>(i) = node->dim(i).value();
  }
  shape_const->shape_status(luci::ShapeStatus::VALID);
  shape_const->name(node->name() + "/shape");
  luci::add_origin(shape_const, luci::get_origin(node));

  reshape->shape(shape_const);

  return reshape;
}

/**
 *  Replace Fully Connected with Batched MatMul
 *
 *  BEFORE
 *
 *         [Node1]         [Node2]
 *           |               |
 *       [transpose]?   [transpose]?
 *               \        /
 *            [FullyConnected]
 *
 *  AFTER
 *
 *              [Node1]  [Node2]
 *                  \      /
 *               [BatchMatMul]
 *                      |
 *                 [Reshape]   [BiasValue]?
 *                        \       /
 *                          [Add]?
 *                            |
 *                       [Activation]?
 *
 * Nodes with "?" denote optional elements
 * NOTE Reshape Op is inserted to keep the original shape of FullyConnected Op
 * Reshape Op can be redundant (input shape == output shape). This can be removed
 * by RemoveUnnecessaryReshapePass.
 */
bool replace_fc_with_matmul(luci::CircleFullyConnected *fc)
{
  luci::CircleNode *x = nullptr;
  luci::CircleNode *y = nullptr;
  luci::CircleTranspose *ty = nullptr;
  luci::CircleTranspose *tx = nullptr;
  bool adj_x = false;
  bool adj_y = true;

  if (dynamic_cast<luci::CircleConst *>(fc->weights()))
    return false; // NonConst

  // NOTE For const inputs, it is possible to block this conversion,
  // because we can make transposed FC rather than matmul to improve
  // accuracy of quantized models by sacrificing latency.
  // See https://github.com/Samsung/ONE/discussions/11941 for more details.

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

  if (x->dtype() != loco::DataType::FLOAT32 || y->dtype() != loco::DataType::FLOAT32)
    return false;

  auto bc = dynamic_cast<luci::CircleConst *>(fc->bias());
  // NOTE bias can be empty as CircleOutputExclude type
  // NOTE we can only handle bias as FLOAT32 type as of now
  if (nullptr != bc && bc->dtype() != loco::DataType::FLOAT32)
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

  auto reshape = create_reshape(fc);
  reshape->tensor(matmul);

  auto all_zero = [](const luci::CircleConst *c) {
    bool ac = true;
    for (uint32_t i = 0; i < c->size<loco::DataType::FLOAT32>() && ac; i++)
    {
      ac &= c->at<loco::DataType::FLOAT32>(i) == 0.0f;
    }
    return ac;
  };

  if (nullptr != bc && !all_zero(bc))
  {
    auto bias_add = fc->graph()->nodes()->create<luci::CircleAdd>();
    bias_add->x(reshape);
    bias_add->y(bc);
    bias_add->name(fc->name() + "/bias_add");
    bias_add->dtype(fc->dtype());
    add_origin(bias_add, get_origin(fc));
    bias_add->fusedActivationFunction(fc->fusedActivationFunction());
    loco::replace(fc).with(bias_add);
  }
  else
  {
    // NOTE bias doesn't exist or bias is all zero
    auto n = fromActivation(reshape, fc->fusedActivationFunction());
    add_origin(n, luci::get_origin(fc));
    n->name(fc->name() + "fusedActivation");
    n->dtype(fc->dtype());
    loco::replace(fc).with(n);
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
