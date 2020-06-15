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

#include "luci/Pass/FuseInstanceNormPassV2.h"
#include "luci/Pass/FuseInstanceNormCommon.h"

#include <cassert>
#include <set>

// Helper to check detail
namespace
{

bool is_instance_mean(luci::CircleMean *mean)
{
  //
  // CHECK 1) input is rank 5
  //
  auto input = mean->input();
  if (not loco::shape_known(input))
    return false;
  auto input_shape = loco::shape_get(input).as<loco::TensorShape>();
  if (input_shape.rank() != 5)
    return false;

  //
  // CHECK 2) 'reduction indices' is CircleConst of value [1,2,4], that is HW of NHWC
  //
  // TODO Support equivalent case, like [-3,-2]
  // TODO Support non-Const case?
  // TODO What if input is NCHW format in Circle?
  auto red_indices = dynamic_cast<luci::CircleConst *>(mean->reduction_indices());
  if (not red_indices)
    return false;
  if (red_indices->rank() != 1)
    return false;
  std::set<int32_t> red_indices_set;
  {
    // TODO Currently only support S32, support other types
    assert(red_indices->dtype() == loco::DataType::S32);
    for (uint32_t i = 0; i < red_indices->dim(0).value(); ++i)
      red_indices_set.insert(red_indices->at<loco::DataType::S32>(i));
  }
  if (red_indices_set.size() != 3)
    return false;
  if (red_indices_set.find(1) == red_indices_set.end())
    return false;
  if (red_indices_set.find(2) == red_indices_set.end())
    return false;
  if (red_indices_set.find(4) == red_indices_set.end())
    return false;

  //
  // CHECK 3) keep_dims == true (?)
  //
  // We only have case of 'keep_dims == true' so far, but it might be okay with 'keep_dims == false'
  // TODO Check this fact, and if true, return true regardless of keep_dims
  return mean->keep_dims();
}

} // namespace

// Helper to fuse Instance Norm
namespace
{

/**
 * SUBGRAPH PATTERN
 *
 *    - Below diagram shows Instance Norm pattern to fuse.
 *    - Execution dependency order is top to the bottom.
 *    - Node name is matched with variable name of InstanceNormPattern class.
 *    - Usually, first word of node name (variable name) is node type. For e.g.
 *      variable 'mean_as_variance' is pointer to TFLMean.
 *    - (Item in parenthesis) means actually exist, but not having a name and
 *      not a variable of InstanceNormPattern class.
 *
 *    TODO support other semantically same patterns for instance norm
 *
 *                 [In]
 *                   |
 *                   V
 *                  ifm
 *                   |
 *                   V
 *     +---------reshape_of_ifm ----+   (reduction indicies)
 *     |             |              |    |
 *     |             |              V    V
 *     |             |       mean_of_reshape -------------+
 *     |             V       |                            |
 *     |           sqdiff <--+   (reduction indicies)     |
 *     |             |             |                      |
 *     |             V             |                      |
 *     |      mean_as_variance <---+  const_as_epsilon    |
 *     |             |                 |                  |
 *     |             V                 |                  |
 *     |      add_as_variance <--------+                  |
 *     |             |                                    |
 *     |             V                                    |
 *     |           rsqrt     const_as_gamma               |
 *     |             |          |                         |
 *     |             V          |                         |
 *     |           mul_gamma <--+                         |
 *     |            |      |                              |
 *     V            V      V                              |
 * mul_as_scaled_reshape   mul_as_scaled_mean <-----------+
 *         |                   |
 *         |   const_as_beta   |
 *         |         |         V
 *         |         +------> sub
 *         V                   |
 *  add_as_terminal <----------+
 *         |
 *         V
 *  reshape_as_terminal
 *         |
 *         V
 *       [Out]
 */
class InstanceNormPattern final
{
public:
  InstanceNormPattern(luci::CircleReshape *candidate)
  {
    assert(candidate);
    reshape_as_terminal = candidate;
  }

public:
  bool matched();
  bool matched() const { return _matched; }

public:
  // Context
  loco::Node *ifm = nullptr;
  luci::CircleReshape *reshape_of_ifm = nullptr;
  luci::CircleMean *mean_of_reshape = nullptr;
  luci::CircleSquaredDifference *sqdiff = nullptr;
  luci::CircleMean *mean_as_variance = nullptr;
  luci::CircleConst *const_as_epsilon = nullptr;
  luci::CircleAdd *add_as_variance = nullptr;
  luci::CircleRsqrt *rsqrt = nullptr;
  luci::CircleConst *const_as_gamma = nullptr;
  luci::CircleMul *mul_gamma = nullptr;
  luci::CircleMul *mul_as_scaled_reshape = nullptr;
  luci::CircleMul *mul_as_scaled_mean = nullptr;
  luci::CircleConst *const_as_beta = nullptr;
  luci::CircleSub *sub = nullptr;
  luci::CircleAdd *add_as_terminal = nullptr;
  luci::CircleReshape *reshape_as_terminal = nullptr;

private:
  bool _matched = false;
};

bool InstanceNormPattern::matched()
{
  if (_matched)
    return true;

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

  // Check order is DFS

  add_as_terminal = dynamic_cast<luci::CircleAdd *>(reshape_as_terminal->tensor());

  CHECK_OR_FALSE(fill(&mul_as_scaled_reshape, &sub).with_commutative_args_of(add_as_terminal));
  CHECK_OR_FALSE(fill(&reshape_of_ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_reshape));

  ifm = reshape_of_ifm->tensor();

  CHECK_OR_FALSE(loco::shape_known(ifm));
  auto ifm_shape = loco::shape_get(ifm);
  CHECK_OR_FALSE(ifm_shape.domain() == loco::Domain::Tensor);
  auto ifm_tensor_shape = ifm_shape.as<loco::TensorShape>();
  CHECK_OR_FALSE(ifm_tensor_shape.rank() == 4);
  uint32_t ifm_channel_depth = ifm_tensor_shape.dim(3).value();

  CHECK_OR_FALSE(fill(&rsqrt, &const_as_gamma).with_commutative_args_of(mul_gamma));
  CHECK_OR_FALSE(is_quasi_1D_with_dummy_dim(const_as_gamma, ifm_channel_depth));

  add_as_variance = dynamic_cast<luci::CircleAdd *>(rsqrt->x());
  CHECK_OR_FALSE(add_as_variance);

  CHECK_OR_FALSE(
      fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));

  CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
  // TODO Support regarding broadcast
  CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

  CHECK_OR_FALSE(is_instance_mean(mean_as_variance));
  sqdiff = dynamic_cast<luci::CircleSquaredDifference *>(mean_as_variance->input());
  CHECK_OR_FALSE(sqdiff);

  loco::Node *reshape_should_be = nullptr;
  CHECK_OR_FALSE(fill(&reshape_should_be, &mean_of_reshape).with_commutative_args_of(sqdiff));
  CHECK_OR_FALSE(reshape_of_ifm == reshape_should_be);
  CHECK_OR_FALSE(is_instance_mean(mean_of_reshape));
  CHECK_OR_FALSE(reshape_of_ifm == mean_of_reshape->input());

  const_as_beta = dynamic_cast<luci::CircleConst *>(sub->x());
  CHECK_OR_FALSE(const_as_beta);
  CHECK_OR_FALSE(is_quasi_1D_with_dummy_dim(const_as_beta, ifm_channel_depth));

  mul_as_scaled_mean = dynamic_cast<luci::CircleMul *>(sub->y());
  CHECK_OR_FALSE(mul_as_scaled_mean);

  luci::CircleMul *mul_gamma_should_be = nullptr;
  luci::CircleMean *mean_of_reshape_should_be = nullptr;
  CHECK_OR_FALSE(fill(&mul_gamma_should_be, &mean_of_reshape_should_be)
                     .with_commutative_args_of(mul_as_scaled_mean));
  CHECK_OR_FALSE(mul_gamma == mul_gamma_should_be);
  CHECK_OR_FALSE(mean_of_reshape == mean_of_reshape_should_be);
#undef CHECK_OR_FALSE
  _matched = true;
  return true;
}

/**
 * Instance norm pattern would be fused like following diagram:
 *
 *    [In] --------------------------- CircleInstanceNorm --- [Out]
 *                                     / /
 *    const_as_gamma --- TFLReshape --- /
 *                                     /
 *    const_as_beta ---- TFLReshape ---
 *
 * Note
 *  - 'const_as_gamma' and 'const_as_beta' are from original graph
 *  - Value of 'const_as_epsilon' would be copied to CircleInstanceNorm's attribute
 *  - TFLReshape is added as CircleInstanceNorm only accept 1D tensor
 *  - 'CircleConst --- TFLReshape' is expected to be fused in constant folding for Reshape
 */
void fuse_instance_norm(const InstanceNormPattern &p)
{
  assert(p.matched());

  auto graph = p.reshape_as_terminal->graph();

  // Make reshape for gamma & beta
  auto reshape_gamma = graph->nodes()->create<luci::CircleReshape>();
  auto reshape_beta = graph->nodes()->create<luci::CircleReshape>();
  {
    auto ifm_shape = loco::shape_get(p.reshape_of_ifm->tensor()).as<loco::TensorShape>();
    uint32_t ifm_channel_depth = ifm_shape.dim(3).value();

    int32_t new_shape[1] = {static_cast<int32_t>(ifm_channel_depth)};

    reshape_gamma->tensor(p.const_as_gamma);
    reshape_beta->tensor(p.const_as_beta);

    luci::set_new_shape(reshape_gamma, new_shape, 1);
    luci::set_new_shape(reshape_beta, new_shape, 1);
  }

  // Make Instance Norm to replace
  auto instance_norm = graph->nodes()->create<luci::CircleInstanceNorm>();
  instance_norm->input(p.ifm);
  instance_norm->gamma(reshape_gamma);
  instance_norm->beta(reshape_beta);
  float epsilon = p.const_as_epsilon->at<loco::DataType::FLOAT32>(0);
  instance_norm->epsilon(epsilon);
  instance_norm->fusedActivationFunction(p.add_as_terminal->fusedActivationFunction());

  replace(p.add_as_terminal).with(instance_norm);
}

} // namespace

namespace luci
{

bool FuseInstanceNormPassV2::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
      continue;

    auto add_candidate = dynamic_cast<luci::CircleAdd *>(reshape->tensor());
    if (not add_candidate)
      continue;

    InstanceNormPattern pattern(reshape);
    if (not pattern.matched())
      continue;

    fuse_instance_norm(pattern);
    changed = true;
  }

  return changed;
}

} // namespace luci
