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

#include "luci/Pass/FuseInstanceNormPass.h"
#include "helpers/NodeFiller.h"
#include "FuseInstanceNormPassInternal.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>

#include <cassert>
#include <set>

// Helper to check detail

/// @return true  When node has shape of '1 x .. x 1 x depth'
bool is_1D_with_dummy_dim(luci::CircleConst *node, uint32_t depth)
{
  auto rank = node->rank();
  uint32_t axis;
  for (axis = 0; axis < rank - 1; ++axis)
  {
    if (node->dim(axis).value() != 1)
      return false;
  }
  return node->dim(axis).value() == depth;
}

/// @return true if node shape consists of ones, except the one before the last dim: 1,...1,depth,1
bool is_quasi_1D_with_dummy_dim(luci::CircleConst *node, uint32_t depth)
{
  auto rank = node->rank();
  // minimal accepted shape is [1 x depth x 1]
  if (rank < 3)
    return false;
  const auto depth_axis = rank - 2;
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    if (axis != depth_axis && node->dim(axis).value() != 1)
      return false;
  }
  return node->dim(depth_axis).value() == depth;
}

bool is_instance_mean_v0(luci::CircleMean *mean)
{
  //
  // CHECK 1) input is rank 4
  //
  auto input = loco::must_cast<luci::CircleNode *>(mean->input());
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;
  if (input->rank() != 4)
    return false;

  //
  // CHECK 2) 'reduction indices' is CircleConst of value [1,2], that is HW of NHWC
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
  if (red_indices_set.size() != 2)
    return false;
  if (red_indices_set.find(1) == red_indices_set.end())
    return false;
  if (red_indices_set.find(2) == red_indices_set.end())
    return false;

  //
  // CHECK 3) keep_dims == true (?)
  //
  // We only have case of 'keep_dims == true' so far, but it might be okay with 'keep_dims == false'
  // TODO Check this fact, and if true, return true regardless of keep_dims
  return mean->keep_dims();
}

bool is_instance_mean_v1(luci::CircleMean *mean)
{
  //
  // CHECK 1) input is rank 5 (NHWCX)
  //
  auto input = loco::must_cast<luci::CircleNode *>(mean->input());
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;
  if (input->rank() != 5)
    return false;

  //
  // CHECK 2) 'reduction indices' is CircleConst of value [1,2,4], that is HWX of NHWCX input shape
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

  // TODO Currently only support S32, support other types
  if (red_indices->dtype() != loco::DataType::S32)
    return false;
  for (uint32_t i = 0; i < red_indices->dim(0).value(); ++i)
    red_indices_set.insert(red_indices->at<loco::DataType::S32>(i));

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

/// @return true  When node has the shape of 1D channel_size
bool is_1D_float32_const(const luci::CircleConst *node, uint32_t channel_size)
{
  if (node->rank() != 1)
    return false;

  if (node->dim(0).value() != channel_size)
    return false;

  if (node->dtype() != loco::DataType::FLOAT32)
    return false;

  if (node->size<loco::DataType::FLOAT32>() != channel_size)
    return false;

  return true;
}

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
 *     +----------- ifm -----+   (reduction indicies)
 *     |             |       |       |
 *     |             |       V       V
 *     |             |      mean_of_ifm ----------------+
 *     |             V       |                          |
 *     |           sqdiff <--+   (reduction indicies)   |
 *     |             |             |                    |
 *     |             V             |                    |
 *     |      mean_as_variance <---+  const_as_epsilon  |
 *     |             |                 |                |
 *     |             V                 |                |
 *     |      add_as_variance <--------+                |
 *     |             |                                  |
 *     |             V                                  |
 *     |           rsqrt   const_as_gamma               |
 *     |             |        |                         |
 *     |             V        |                         |
 *     |         mul_gamma <--+                         |
 *     |          |     |                               |
 *     V          V     V                               |
 * mul_as_scaled_ifm   mul_as_scaled_mean <-------------+
 *         |                   |
 *         |   const_as_beta   |
 *         |         |         V
 *         |         +------> sub
 *         V                   |
 *  add_as_terminal <----------+
 *         |
 *         V
 *       [Out]
 *-------------------------------------------------------------------
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
 *-------------------------------------------------------------------
 *                          [In]
 *                            |
 *                            V
 *      +----+-------------- ifm
 *      |    |  (reduction    |
 *      |    |   indicies)    |
 *      |    |     |          |
 *      |    V     V          |
 *      |  mean_of_ifm        |
 *      |       |             V
 *      |       +------->  sqdiff   (reduction indicies)
 *      V       |             |            |
 *     sub <----+             V            |
 *      |             mean_as_variance <---+  const_as_epsilon
 *      |                     |                     |
 *      |                     V                     |
 *      |              add_as_variance <------------+
 *      |                     |    (0.5)
 *      |                     V      |
 *      |                    pow <---+
 *      |                     |
 *      V                     |
 *     div <------------------+    const_as_gamma
 *      |                              |
 *      V                              |
 *    mul_gamma <----------------------+
 *      |                const_as_beta
 *      V                     |
 *   add_as_terminal <--------+
 *       |
 *       V
 *     [Out]
 */
class InstanceNormPattern final
{
public:
  enum PatternVersion
  {
    Version_0,
    Version_1,
    Version_2,
  };

  InstanceNormPattern(luci::CircleAdd *candidate, PatternVersion pv)
  {
    assert(candidate);
    add_as_terminal = candidate;
    _pv = pv;
  }

public:
  bool matched();
  bool matched() const { return _matched; }

  PatternVersion version() const { return _pv; }

public:
  // Context
  loco::Node *ifm = nullptr;
  luci::CircleReshape *reshape_of_ifm = nullptr;
  luci::CircleMean *mean_of_ifm = nullptr;
  luci::CircleMean *mean_of_reshape = nullptr;
  luci::CircleSquaredDifference *sqdiff = nullptr;
  luci::CircleMean *mean_as_variance = nullptr;
  luci::CircleConst *const_as_epsilon = nullptr;
  luci::CircleAdd *add_as_variance = nullptr;
  luci::CircleRsqrt *rsqrt = nullptr;
  luci::CircleConst *const_as_gamma = nullptr;
  luci::CircleMul *mul_gamma = nullptr;
  luci::CircleMul *mul_as_scaled_ifm = nullptr;
  luci::CircleMul *mul_as_scaled_mean = nullptr;
  luci::CircleMul *mul_as_scaled_reshape = nullptr;
  luci::CircleConst *const_as_beta = nullptr;
  luci::CircleSub *sub = nullptr;
  luci::CircleAdd *add_as_terminal = nullptr;
  luci::CirclePow *pow = nullptr;
  luci::CircleDiv *div = nullptr;

private:
  bool _matched = false;
  PatternVersion _pv;
};

bool InstanceNormPattern::matched()
{
  if (_matched)
    return true;

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

  // Check order is DFS

  // Version 2 is quite different from Version 0 and 1.
  // So it is handled in the separate if statement
  if (_pv == PatternVersion::Version_2)
  {
    CHECK_OR_FALSE(
      luci::fill(&mul_gamma, &const_as_beta).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(luci::fill(&div, &const_as_gamma).with_commutative_args_of(mul_gamma));

    sub = dynamic_cast<luci::CircleSub *>(div->x());
    CHECK_OR_FALSE(sub);

    ifm = sub->x();
    CHECK_OR_FALSE(ifm);

    luci::CircleNode *ifm_node = loco::must_cast<luci::CircleNode *>(ifm);
    CHECK_OR_FALSE(ifm_node->rank() == 4);
    CHECK_OR_FALSE(ifm_node->dim(3).known());
    uint32_t ifm_channel_depth = ifm_node->dim(3).value();

    mean_of_ifm = dynamic_cast<luci::CircleMean *>(sub->y());
    CHECK_OR_FALSE(mean_of_ifm);

    CHECK_OR_FALSE(ifm == mean_of_ifm->input());

    pow = dynamic_cast<luci::CirclePow *>(div->y());
    CHECK_OR_FALSE(pow);

    add_as_variance = dynamic_cast<luci::CircleAdd *>(pow->x());
    CHECK_OR_FALSE(add_as_variance);

    luci::CircleConst *zero_point_five = dynamic_cast<luci::CircleConst *>(pow->y());
    CHECK_OR_FALSE(zero_point_five);
    CHECK_OR_FALSE(zero_point_five->dtype() == loco::DataType::FLOAT32);
    // TODO Support regarding broadcast
    CHECK_OR_FALSE(zero_point_five->size<loco::DataType::FLOAT32>() == 1);
    CHECK_OR_FALSE(zero_point_five->at<loco::DataType::FLOAT32>(0) == 0.5);

    CHECK_OR_FALSE(
      luci::fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));
    CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
    // TODO Support regarding broadcast
    CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

    CHECK_OR_FALSE(is_instance_mean_v0(mean_as_variance));

    sqdiff = dynamic_cast<luci::CircleSquaredDifference *>(mean_as_variance->input());
    CHECK_OR_FALSE(sqdiff);

    loco::Node *ifm_should_be = nullptr;
    luci::CircleMean *mean_of_ifm_should_be = nullptr;
    CHECK_OR_FALSE(
      luci::fill(&ifm_should_be, &mean_of_ifm_should_be).with_commutative_args_of(sqdiff));
    CHECK_OR_FALSE(ifm == ifm_should_be);
    CHECK_OR_FALSE(mean_of_ifm == mean_of_ifm_should_be);

    // Check for channel size
    CHECK_OR_FALSE(is_1D_float32_const(const_as_gamma, ifm_channel_depth));
    CHECK_OR_FALSE(is_1D_float32_const(const_as_beta, ifm_channel_depth));

    _matched = true;
    return true;
  }

  if (_pv == PatternVersion::Version_0)
  {
    CHECK_OR_FALSE(luci::fill(&mul_as_scaled_ifm, &sub).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(luci::fill(&ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_ifm));
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(
      luci::fill(&mul_as_scaled_reshape, &sub).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(
      luci::fill(&reshape_of_ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_reshape));
    ifm = reshape_of_ifm->tensor();
  }

  auto ifm_circle = loco::must_cast<luci::CircleNode *>(ifm);
  CHECK_OR_FALSE(ifm_circle->shape_status() == luci::ShapeStatus::VALID);
  CHECK_OR_FALSE(ifm_circle->rank() == 4);
  CHECK_OR_FALSE(ifm_circle->dim(3).known());
  uint32_t ifm_channel_depth = ifm_circle->dim(3).value();

  CHECK_OR_FALSE(luci::fill(&rsqrt, &const_as_gamma).with_commutative_args_of(mul_gamma));

  if (_pv == PatternVersion::Version_0)
  {
    CHECK_OR_FALSE(is_1D_with_dummy_dim(const_as_gamma, ifm_channel_depth));
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(is_quasi_1D_with_dummy_dim(const_as_gamma, ifm_channel_depth));
  }

  add_as_variance = dynamic_cast<luci::CircleAdd *>(rsqrt->x());
  CHECK_OR_FALSE(add_as_variance);

  CHECK_OR_FALSE(
    luci::fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));

  CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
  // TODO Support regarding broadcast
  CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

  if (_pv == PatternVersion::Version_0)
  {
    CHECK_OR_FALSE(is_instance_mean_v0(mean_as_variance));
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(is_instance_mean_v1(mean_as_variance));
  }

  sqdiff = dynamic_cast<luci::CircleSquaredDifference *>(mean_as_variance->input());
  CHECK_OR_FALSE(sqdiff);

  if (_pv == PatternVersion::Version_0)
  {
    loco::Node *ifm_should_be = nullptr;
    CHECK_OR_FALSE(luci::fill(&ifm_should_be, &mean_of_ifm).with_commutative_args_of(sqdiff));
    CHECK_OR_FALSE(ifm == ifm_should_be);
    CHECK_OR_FALSE(is_instance_mean_v0(mean_of_ifm));
    CHECK_OR_FALSE(ifm == mean_of_ifm->input());
  }
  if (_pv == PatternVersion::Version_1)
  {
    loco::Node *reshape_should_be = nullptr;
    CHECK_OR_FALSE(
      luci::fill(&reshape_should_be, &mean_of_reshape).with_commutative_args_of(sqdiff));
    CHECK_OR_FALSE(reshape_of_ifm == reshape_should_be);
    CHECK_OR_FALSE(is_instance_mean_v1(mean_of_reshape));
    CHECK_OR_FALSE(reshape_of_ifm == mean_of_reshape->input());
  }

  const_as_beta = dynamic_cast<luci::CircleConst *>(sub->x());
  CHECK_OR_FALSE(const_as_beta);

  if (_pv == PatternVersion::Version_0)
  {
    CHECK_OR_FALSE(is_1D_with_dummy_dim(const_as_beta, ifm_channel_depth));
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(is_quasi_1D_with_dummy_dim(const_as_beta, ifm_channel_depth));
  }

  mul_as_scaled_mean = dynamic_cast<luci::CircleMul *>(sub->y());
  CHECK_OR_FALSE(mul_as_scaled_mean);

  luci::CircleMul *mul_gamma_should_be = nullptr;
  luci::CircleMean *mean_of_ifm_should_be = nullptr;
  luci::CircleMean *mean_of_reshape_should_be = nullptr;

  if (_pv == PatternVersion::Version_0)
  {
    CHECK_OR_FALSE(luci::fill(&mul_gamma_should_be, &mean_of_ifm_should_be)
                     .with_commutative_args_of(mul_as_scaled_mean));
    CHECK_OR_FALSE(mul_gamma == mul_gamma_should_be);
    CHECK_OR_FALSE(mean_of_ifm == mean_of_ifm_should_be);
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(luci::fill(&mul_gamma_should_be, &mean_of_reshape_should_be)
                     .with_commutative_args_of(mul_as_scaled_mean));
    CHECK_OR_FALSE(mul_gamma == mul_gamma_should_be);
    CHECK_OR_FALSE(mean_of_reshape == mean_of_reshape_should_be);
  }

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

  auto graph = p.add_as_terminal->graph();

  // Version 0 and 1 need to reshape
  if (p.version() != InstanceNormPattern::Version_2)
  {
    p.const_as_gamma->rank(1);
    p.const_as_gamma->dim(0).set(p.const_as_gamma->size<loco::DataType::FLOAT32>());
    p.const_as_beta->rank(1);
    p.const_as_beta->dim(0).set(p.const_as_beta->size<loco::DataType::FLOAT32>());

    p.const_as_gamma->shape_status(luci::ShapeStatus::UNDEFINED);
    p.const_as_beta->shape_status(luci::ShapeStatus::UNDEFINED);
  }

  // Make Instance Norm to replace
  auto instance_norm = graph->nodes()->create<luci::CircleInstanceNorm>();
  instance_norm->input(p.ifm);
  instance_norm->gamma(p.const_as_gamma);
  instance_norm->beta(p.const_as_beta);
  float epsilon = p.const_as_epsilon->at<loco::DataType::FLOAT32>(0);
  instance_norm->epsilon(epsilon);
  instance_norm->fusedActivationFunction(p.add_as_terminal->fusedActivationFunction());
  // NOTE unique name should be assigned in export
  instance_norm->name("InstanceNorm");

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(p.sqdiff),
    luci::get_origin(p.mean_as_variance),
    luci::get_origin(p.add_as_variance),
    luci::get_origin(p.mul_gamma),
    luci::get_origin(p.sub),
    luci::get_origin(p.add_as_terminal)};
  if (p.version() == InstanceNormPattern::PatternVersion::Version_0)
  {
    origin_vec.push_back(luci::get_origin(p.mean_of_ifm));
    origin_vec.push_back(luci::get_origin(p.rsqrt));
    origin_vec.push_back(luci::get_origin(p.mul_as_scaled_ifm));
    origin_vec.push_back(luci::get_origin(p.mul_as_scaled_mean));
  }
  if (p.version() == InstanceNormPattern::PatternVersion::Version_1)
  {
    origin_vec.push_back(luci::get_origin(p.reshape_of_ifm));
    origin_vec.push_back(luci::get_origin(p.mean_of_reshape));
    origin_vec.push_back(luci::get_origin(p.rsqrt));
    origin_vec.push_back(luci::get_origin(p.mul_as_scaled_mean));
    origin_vec.push_back(luci::get_origin(p.mul_as_scaled_reshape));
  }
  if (p.version() == InstanceNormPattern::PatternVersion::Version_2)
  {
    origin_vec.push_back(luci::get_origin(p.mean_of_ifm));
    origin_vec.push_back(luci::get_origin(p.pow));
    origin_vec.push_back(luci::get_origin(p.div));
  }
  luci::add_origin(instance_norm, luci::composite_origin(origin_vec));

  replace(p.add_as_terminal).with(instance_norm);
}

} // namespace

namespace
{

bool is_add_input_mul_const(luci::CircleAdd *add)
{
  luci::CircleMul *p_mul = nullptr;
  luci::CircleConst *p_const = nullptr;

  return luci::fill(&p_mul, &p_const).with_commutative_args_of(add);
}

} // namespace

namespace luci
{

bool FuseInstanceNormPass::run(loco::Graph *g)
{
  bool changed = false;
  luci::CircleAdd *add;
  InstanceNormPattern::PatternVersion pv;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
    {
      add = dynamic_cast<luci::CircleAdd *>(node);
      if (not add)
        continue;
      pv = InstanceNormPattern::PatternVersion::Version_0;

      if (is_add_input_mul_const(add))
        pv = InstanceNormPattern::PatternVersion::Version_2;
    }
    else
    {
      add = dynamic_cast<luci::CircleAdd *>(reshape->tensor());
      if (not add)
        continue;
      pv = InstanceNormPattern::PatternVersion::Version_1;
    }

    InstanceNormPattern pattern(add, pv);
    if (not pattern.matched())
      continue;

    fuse_instance_norm(pattern);
    changed = true;
  }

  return changed;
}

} // namespace luci
