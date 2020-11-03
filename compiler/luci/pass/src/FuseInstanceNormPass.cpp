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
#include "FuseInstanceNormPassInternal.h"

#include <luci/IR/CircleNodes.h>

#include <loco/Service/ShapeInference.h>

#include <cassert>
#include <set>

// Helper to find commutative node's arguments
namespace
{

/**
 * INTRODUCTION
 *         Binary operation f(x,y) is 'commutative' when
 *         f(x,y) == f(y,x) holds for all x, y.
 *         For examples, ADD, MUL and SQUARED_DIFFERENCE are commutative.
 *         These helpers make it easy to find commutative arguemnts of commtative node.
 *
 * HOW TO USE
 *         COMM_NODE *node;
 *         ARG_TYPE_1 *arg1;
 *         ARG_TYPE_2 *arg2;
 *
 *         bool ok = fill(&arg1, &arg2).with_commutative_args_of(node);
 *
 * Result
 *         If 'node's commutative argument types are actually {ARG_TYPE_1, ARG_TYPE_2}
 *         (as a set), 'arg1' and 'arg2' set as actual 'node's arguemnts with matching
 *         type, and return value 'ok' is true.
 *         Otherwise, 'arg1' and 'arg2' not changed, 'ok' is false.
 */

template <class ARG_TYPE_1, class ARG_TYPE_2> class NodeFiller final
{
public:
  NodeFiller(ARG_TYPE_1 **arg_1, ARG_TYPE_2 **arg_2) : _arg_1(arg_1), _arg_2(arg_2)
  {
    // DO NOTHING
  }

  /**
   * @return true   When 'node's argument types are 'ARG_TYPE_1' and 'ARG_TYPE_2'
   *                In such case, it assign '_arg_1' and '_arg_2' to actual arguments
   *
   * @return false  When 'node's argument types are NOT matched with 'ARG_TYPE_*'
   *                In such case, it does not amend '_arg_1' and '_arg_2'
   *
   * @require       COMM_NODE has member x() and y()
   */
  template <class COMM_NODE> bool with_commutative_args_of(const COMM_NODE *node);

private:
  ARG_TYPE_1 **_arg_1;
  ARG_TYPE_2 **_arg_2;
};

template <class ARG_TYPE_1, class ARG_TYPE_2>
inline NodeFiller<ARG_TYPE_1, ARG_TYPE_2> fill(ARG_TYPE_1 **arg_1, ARG_TYPE_2 **arg_2)
{
  return NodeFiller<ARG_TYPE_1, ARG_TYPE_2>{arg_1, arg_2};
}

template <class ARG_TYPE_1, class ARG_TYPE_2>
template <class COMM_NODE>
bool NodeFiller<ARG_TYPE_1, ARG_TYPE_2>::with_commutative_args_of(const COMM_NODE *node)
{
  // Case 1) X == ARG_TYPE_1 / Y == ARG_TYPE_2
  {
    auto x = dynamic_cast<ARG_TYPE_1 *>(node->x());
    auto y = dynamic_cast<ARG_TYPE_2 *>(node->y());

    if (x && y)
    {
      *_arg_1 = x;
      *_arg_2 = y;
      return true;
    }
  }

  // Case 2) X == ARG_TYPE_2 / Y == ARG_TYPE_1
  {
    auto x = dynamic_cast<ARG_TYPE_2 *>(node->x());
    auto y = dynamic_cast<ARG_TYPE_1 *>(node->y());

    if (x && y)
    {
      *_arg_1 = y;
      *_arg_2 = x;
      return true;
    }
  }

  return false;
}

} // namespace

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
  auto input = mean->input();
  if (not loco::shape_known(input))
    return false;
  auto input_shape = loco::shape_get(input).as<loco::TensorShape>();
  if (input_shape.rank() != 4)
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
  auto input = mean->input();
  if (not loco::shape_known(input))
    return false;
  auto input_shape = loco::shape_get(input).as<loco::TensorShape>();
  if (input_shape.rank() != 5)
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
    CHECK_OR_FALSE(fill(&mul_gamma, &const_as_beta).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(fill(&div, &const_as_gamma).with_commutative_args_of(mul_gamma));

    sub = dynamic_cast<luci::CircleSub *>(div->x());
    CHECK_OR_FALSE(sub);

    ifm = sub->x();
    CHECK_OR_FALSE(ifm);

    luci::CircleNode *ifm_node = loco::must_cast<luci::CircleNode *>(ifm);
    CHECK_OR_FALSE(ifm_node->rank() == 4);
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
        fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));
    CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
    // TODO Support regarding broadcast
    CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

    CHECK_OR_FALSE(is_instance_mean_v0(mean_as_variance));

    sqdiff = dynamic_cast<luci::CircleSquaredDifference *>(mean_as_variance->input());
    CHECK_OR_FALSE(sqdiff);

    loco::Node *ifm_should_be = nullptr;
    luci::CircleMean *mean_of_ifm_should_be = nullptr;
    CHECK_OR_FALSE(fill(&ifm_should_be, &mean_of_ifm_should_be).with_commutative_args_of(sqdiff));
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
    CHECK_OR_FALSE(fill(&mul_as_scaled_ifm, &sub).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(fill(&ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_ifm));
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(fill(&mul_as_scaled_reshape, &sub).with_commutative_args_of(add_as_terminal));
    CHECK_OR_FALSE(
        fill(&reshape_of_ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_reshape));
    ifm = reshape_of_ifm->tensor();
  }

  CHECK_OR_FALSE(loco::shape_known(ifm));
  auto ifm_shape = loco::shape_get(ifm);
  CHECK_OR_FALSE(ifm_shape.domain() == loco::Domain::Tensor);
  auto ifm_tensor_shape = ifm_shape.as<loco::TensorShape>();
  CHECK_OR_FALSE(ifm_tensor_shape.rank() == 4);
  uint32_t ifm_channel_depth = ifm_tensor_shape.dim(3).value();

  CHECK_OR_FALSE(fill(&rsqrt, &const_as_gamma).with_commutative_args_of(mul_gamma));

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
      fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));

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
    CHECK_OR_FALSE(fill(&ifm_should_be, &mean_of_ifm).with_commutative_args_of(sqdiff));
    CHECK_OR_FALSE(ifm == ifm_should_be);
    CHECK_OR_FALSE(is_instance_mean_v0(mean_of_ifm));
    CHECK_OR_FALSE(ifm == mean_of_ifm->input());
  }
  if (_pv == PatternVersion::Version_1)
  {
    loco::Node *reshape_should_be = nullptr;
    CHECK_OR_FALSE(fill(&reshape_should_be, &mean_of_reshape).with_commutative_args_of(sqdiff));
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
    CHECK_OR_FALSE(fill(&mul_gamma_should_be, &mean_of_ifm_should_be)
                       .with_commutative_args_of(mul_as_scaled_mean));
    CHECK_OR_FALSE(mul_gamma == mul_gamma_should_be);
    CHECK_OR_FALSE(mean_of_ifm == mean_of_ifm_should_be);
  }
  if (_pv == PatternVersion::Version_1)
  {
    CHECK_OR_FALSE(fill(&mul_gamma_should_be, &mean_of_reshape_should_be)
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

  // Special case for version 2 (no need to reshape)
  if (p.version() == InstanceNormPattern::Version_2)
  {
    // Make Instance Norm to replace
    auto instance_norm = graph->nodes()->create<luci::CircleInstanceNorm>();
    instance_norm->input(p.ifm);
    instance_norm->gamma(p.const_as_gamma);
    instance_norm->beta(p.const_as_beta);
    float epsilon = p.const_as_epsilon->at<loco::DataType::FLOAT32>(0);
    instance_norm->epsilon(epsilon);
    instance_norm->fusedActivationFunction(p.add_as_terminal->fusedActivationFunction());

    replace(p.add_as_terminal).with(instance_norm);

    return;
  }

  // Make reshape for gamma & beta
  auto reshape_gamma = graph->nodes()->create<luci::CircleReshape>();
  auto reshape_beta = graph->nodes()->create<luci::CircleReshape>();
  {
    auto ifm_shape = loco::shape_get(p.ifm).as<loco::TensorShape>();
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

      auto x = loco::must_cast<luci::CircleNode *>(add->x());
      auto y = loco::must_cast<luci::CircleNode *>(add->y());
      if ((x->opcode() == luci::CircleOpcode::MUL &&
           y->opcode() == luci::CircleOpcode::CIRCLECONST) ||
          (x->opcode() == luci::CircleOpcode::CIRCLECONST &&
           y->opcode() == luci::CircleOpcode::MUL))
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
