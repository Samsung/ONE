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

bool is_instance_mean_v1(luci::CircleMean *mean)
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
 * Version_1
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
 * Version_2
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
 *-------------------------------------------------------------------
 * Version_3
 *                          [In]
 *                            |
 *                            V
 *      +----+-------------- ifm ---+
 *      |    |  (reduction    |     |  (reduction
 *      |    |   indicies)    |     |   indicies)
 *      |    |     |          |     |     |
 *      |    V     V          |     V     V
 *      |  mean_of_ifm        |   mean_of_ifm_2
 *      |       |             |       |
 *      V       |             V       |
 *     sub <----+           sub_2 <---+
 *      |                     |
 *      |                     V
 *      |                   square
 *      |                     |   (reduction indicies)
 *      |                     |            |
 *      |                     V            |
 *      |             mean_as_variance <---+
 *      |                     |
 *      |                     V
 *      |                    sqrt    const_as_epsilon
 *      |                     |            |
 *      |                     V            |
 *      |              add_as_variance <---+
 *      |                     |
 *      V                     |
 *     div <------------------+    const_as_gamma
 *      |                              |
 *      V                              |
 *    mul_gamma <----------------------+
 *      |                const_as_beta
 *      V                     |
 *   add_as_terminal <--------+
 *      |
 *      V
 *    [Out]
 *-------------------------------------------------------------------
 * Version_4
 * - mul_gamma and add_as_terminal are removed for const_as_gamma = 1.0
 *   and const_as_beta = 0.0
 *                          [In]
 *                            |
 *                            V
 *      +----+-------------- ifm ---+
 *      |    |  (reduction    |     |  (reduction
 *      |    |   indicies)    |     |   indicies)
 *      |    |     |          |     |     |
 *      |    V     V          |     V     V
 *      |  mean_of_ifm        |   mean_of_ifm_2
 *      |       |             |       |
 *      V       |             V       |
 *     sub <----+           sub_2 <---+
 *      |                     |
 *      |                     V
 *      |                   square
 *      |                     |   (reduction indicies)
 *      |                     |            |
 *      |                     V            |
 *      |             mean_as_variance <---+
 *      |                     |
 *      |                     V
 *      |                    sqrt    const_as_epsilon
 *      |                     |            |
 *      |                     V            |
 *      |              add_as_variance <---+
 *      |                     |
 *      V                     |
 *     div <------------------+
 *      |
 *      V
 *    [Out]
 */
class InstanceNormPattern final
{
public:
  enum PatternVersion
  {
    Version_Unknown,
    Version_1,
    Version_2,
    Version_3,
    Version_4,
  };

  InstanceNormPattern(luci::CircleAdd *candidate, PatternVersion pv)
  {
    assert(candidate);
    add_as_terminal = candidate;
    _pv = pv;
  }

  InstanceNormPattern(luci::CircleDiv *candidate, PatternVersion pv)
  {
    assert(candidate);
    div = candidate;
    _pv = pv;
  }

private:
  template <enum PatternVersion> bool match();

public:
  bool matched();
  bool matched() const { return _matched; }

  PatternVersion version() const { return _pv; }

public:
  // Context
  loco::Node *ifm = nullptr;
  luci::CircleReshape *reshape_of_ifm = nullptr;
  luci::CircleMean *mean_of_ifm = nullptr;
  luci::CircleMean *mean_of_ifm_2 = nullptr;
  luci::CircleMean *mean_of_reshape = nullptr;
  luci::CircleSquaredDifference *sqdiff = nullptr;
  luci::CircleSquare *square = nullptr;
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
  luci::CircleSub *sub_2 = nullptr;
  luci::CircleAdd *add_as_terminal = nullptr;
  luci::CirclePow *pow = nullptr;
  luci::CircleSqrt *sqrt = nullptr;
  luci::CircleDiv *div = nullptr;

private:
  bool _matched = false;
  PatternVersion _pv;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

template <> bool InstanceNormPattern::match<InstanceNormPattern::PatternVersion::Version_1>()
{
  CHECK_OR_FALSE(luci::fill(&mul_as_scaled_ifm, &sub).with_commutative_args_of(add_as_terminal));
  CHECK_OR_FALSE(luci::fill(&ifm, &mul_gamma).with_commutative_args_of(mul_as_scaled_ifm));

  auto ifm_circle = loco::must_cast<luci::CircleNode *>(ifm);
  CHECK_OR_FALSE(ifm_circle->shape_status() == luci::ShapeStatus::VALID);
  CHECK_OR_FALSE(ifm_circle->rank() == 4);
  CHECK_OR_FALSE(ifm_circle->dim(3).known());
  uint32_t ifm_channel_depth = ifm_circle->dim(3).value();

  CHECK_OR_FALSE(luci::fill(&rsqrt, &const_as_gamma).with_commutative_args_of(mul_gamma));

  CHECK_OR_FALSE(is_1D_with_dummy_dim(const_as_gamma, ifm_channel_depth));

  add_as_variance = dynamic_cast<luci::CircleAdd *>(rsqrt->x());
  CHECK_OR_FALSE(add_as_variance);

  CHECK_OR_FALSE(
    luci::fill(&mean_as_variance, &const_as_epsilon).with_commutative_args_of(add_as_variance));

  CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
  // TODO Support regarding broadcast
  CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

  CHECK_OR_FALSE(is_instance_mean_v1(mean_as_variance));

  sqdiff = dynamic_cast<luci::CircleSquaredDifference *>(mean_as_variance->input());
  CHECK_OR_FALSE(sqdiff);

  loco::Node *ifm_should_be = nullptr;
  CHECK_OR_FALSE(luci::fill(&ifm_should_be, &mean_of_ifm).with_commutative_args_of(sqdiff));
  CHECK_OR_FALSE(ifm == ifm_should_be);
  CHECK_OR_FALSE(is_instance_mean_v1(mean_of_ifm));
  CHECK_OR_FALSE(ifm == mean_of_ifm->input());

  const_as_beta = dynamic_cast<luci::CircleConst *>(sub->x());
  CHECK_OR_FALSE(const_as_beta);
  CHECK_OR_FALSE(is_1D_with_dummy_dim(const_as_beta, ifm_channel_depth));

  luci::CircleMul *mul_gamma_should_be = nullptr;
  luci::CircleMean *mean_of_ifm_should_be = nullptr;

  mul_as_scaled_mean = dynamic_cast<luci::CircleMul *>(sub->y());
  CHECK_OR_FALSE(mul_as_scaled_mean);
  CHECK_OR_FALSE(luci::fill(&mul_gamma_should_be, &mean_of_ifm_should_be)
                   .with_commutative_args_of(mul_as_scaled_mean));
  CHECK_OR_FALSE(mul_gamma == mul_gamma_should_be);
  CHECK_OR_FALSE(mean_of_ifm == mean_of_ifm_should_be);

  _matched = true;
  return true;
}

template <> bool InstanceNormPattern::match<InstanceNormPattern::PatternVersion::Version_2>()
{
  CHECK_OR_FALSE(luci::fill(&mul_gamma, &const_as_beta).with_commutative_args_of(add_as_terminal));
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

  CHECK_OR_FALSE(is_instance_mean_v1(mean_as_variance));

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

template <> bool InstanceNormPattern::match<InstanceNormPattern::PatternVersion::Version_3>()
{
  CHECK_OR_FALSE(luci::fill(&mul_gamma, &const_as_beta).with_commutative_args_of(add_as_terminal));
  CHECK_OR_FALSE(luci::fill(&div, &const_as_gamma).with_commutative_args_of(mul_gamma));
  CHECK_OR_FALSE(luci::fill(&sub, &add_as_variance).with_commutative_args_of(div));

  // check left sub
  ifm = sub->x();
  CHECK_OR_FALSE(ifm);

  luci::CircleNode *ifm_node = loco::must_cast<luci::CircleNode *>(ifm);
  CHECK_OR_FALSE(ifm_node->rank() == 4);
  CHECK_OR_FALSE(ifm_node->dim(3).known());

  mean_of_ifm = dynamic_cast<luci::CircleMean *>(sub->y());
  CHECK_OR_FALSE(mean_of_ifm);
  CHECK_OR_FALSE(ifm == mean_of_ifm->input());

  // continue search from add_as_variance
  CHECK_OR_FALSE(luci::fill(&sqrt, &const_as_epsilon).with_commutative_args_of(add_as_variance));
  CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
  // TODO Support regarding broadcast
  CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

  mean_as_variance = dynamic_cast<luci::CircleMean *>(sqrt->x());
  CHECK_OR_FALSE(mean_as_variance);

  square = dynamic_cast<luci::CircleSquare *>(mean_as_variance->input());
  CHECK_OR_FALSE(square);

  sub_2 = dynamic_cast<luci::CircleSub *>(square->x());
  CHECK_OR_FALSE(sub_2);
  CHECK_OR_FALSE(ifm == sub_2->x());

  mean_of_ifm_2 = dynamic_cast<luci::CircleMean *>(sub_2->y());
  CHECK_OR_FALSE(mean_of_ifm_2);
  CHECK_OR_FALSE(ifm == mean_of_ifm_2->input());

  loco::Node *ifm_should_be = nullptr;
  luci::CircleMean *mean_of_ifm_2_should_be = nullptr;
  CHECK_OR_FALSE(
    luci::fill(&ifm_should_be, &mean_of_ifm_2_should_be).with_commutative_args_of(sub_2));
  CHECK_OR_FALSE(ifm == ifm_should_be);
  CHECK_OR_FALSE(mean_of_ifm_2 == mean_of_ifm_2_should_be);

  _matched = true;
  return true;
}

luci::CircleConst *make_const_one(loco::Graph *graph, float value)
{
  auto const_one = graph->nodes()->create<luci::CircleConst>();
  const_one->dtype(loco::DataType::FLOAT32);
  const_one->rank(1);
  const_one->size<loco::DataType::FLOAT32>(1);
  const_one->at<loco::DataType::FLOAT32>(0) = value;
  return const_one;
}

template <> bool InstanceNormPattern::match<InstanceNormPattern::PatternVersion::Version_4>()
{
  CHECK_OR_FALSE(div);
  CHECK_OR_FALSE(luci::fill(&sub, &add_as_variance).with_commutative_args_of(div));

  // check left sub
  ifm = sub->x();
  CHECK_OR_FALSE(ifm);

  luci::CircleNode *ifm_node = loco::must_cast<luci::CircleNode *>(ifm);
  CHECK_OR_FALSE(ifm_node->rank() == 4);
  CHECK_OR_FALSE(ifm_node->dim(3).known());

  mean_of_ifm = dynamic_cast<luci::CircleMean *>(sub->y());
  CHECK_OR_FALSE(mean_of_ifm);
  CHECK_OR_FALSE(ifm == mean_of_ifm->input());

  // continue search from add_as_variance
  CHECK_OR_FALSE(luci::fill(&sqrt, &const_as_epsilon).with_commutative_args_of(add_as_variance));
  CHECK_OR_FALSE(const_as_epsilon->dtype() == loco::DataType::FLOAT32);
  // TODO Support regarding broadcast
  CHECK_OR_FALSE(const_as_epsilon->size<loco::DataType::FLOAT32>() == 1);

  mean_as_variance = dynamic_cast<luci::CircleMean *>(sqrt->x());
  CHECK_OR_FALSE(mean_as_variance);

  square = dynamic_cast<luci::CircleSquare *>(mean_as_variance->input());
  CHECK_OR_FALSE(square);

  sub_2 = dynamic_cast<luci::CircleSub *>(square->x());
  CHECK_OR_FALSE(sub_2);
  CHECK_OR_FALSE(ifm == sub_2->x());

  mean_of_ifm_2 = dynamic_cast<luci::CircleMean *>(sub_2->y());
  CHECK_OR_FALSE(mean_of_ifm_2);
  CHECK_OR_FALSE(ifm == mean_of_ifm_2->input());

  loco::Node *ifm_should_be = nullptr;
  luci::CircleMean *mean_of_ifm_2_should_be = nullptr;
  CHECK_OR_FALSE(
    luci::fill(&ifm_should_be, &mean_of_ifm_2_should_be).with_commutative_args_of(sub_2));
  CHECK_OR_FALSE(ifm == ifm_should_be);
  CHECK_OR_FALSE(mean_of_ifm_2 == mean_of_ifm_2_should_be);

  assert(const_as_gamma == nullptr);
  assert(const_as_beta == nullptr);
  assert(mul_gamma == nullptr);
  assert(add_as_terminal == nullptr);

  // create 1.0 gamma and 0.0 beta
  auto graph = div->graph();
  const_as_gamma = make_const_one(graph, 1.0f);
  const_as_beta = make_const_one(graph, 0.0f);
  const_as_gamma->name(div->name() + "/gamma");
  const_as_beta->name(div->name() + "/beta");

  _matched = true;
  return true;
}

bool InstanceNormPattern::matched()
{
  if (_matched)
    return true;

  // Check order is DFS

  switch (_pv)
  {
    case PatternVersion::Version_1:
      return match<PatternVersion::Version_1>();
    case PatternVersion::Version_2:
      return match<PatternVersion::Version_2>();
    case PatternVersion::Version_3:
      return match<PatternVersion::Version_3>();
    case PatternVersion::Version_4:
      return match<PatternVersion::Version_4>();

    default:
      break;
  }

  throw std::runtime_error("Invalid InstanceNorm PatternVersion.");
}

#undef CHECK_OR_FALSE

/**
 * Instance norm pattern would be fused like following diagram:
 *
 *    [In] -------------- CircleInstanceNorm --- [Out]
 *                        / /
 *    const_as_gamma ----  /
 *                        /
 *    const_as_beta -----
 *
 * Note
 *  - 'const_as_gamma' and 'const_as_beta' are from original graph
 *  - Value of 'const_as_epsilon' would be copied to CircleInstanceNorm's attribute
 *  - Two CircleConst shape is updated as CircleInstanceNorm only accept 1D tensor
 *  - 'CircleConst --- TFLReshape' is expected to be fused in constant folding for Reshape
 */

class FuseInstanceNorm final
{
public:
  FuseInstanceNorm(const InstanceNormPattern &p) : _p(p) {}

public:
  void apply(void);

private:
  template <InstanceNormPattern::PatternVersion> void apply(void);

private:
  void reshape_gamma_beta(void);
  luci::CircleInstanceNorm *create_inst_norm(loco::Graph *graph);

private:
  const InstanceNormPattern &_p;
};

void FuseInstanceNorm::reshape_gamma_beta()
{
  // Version 1 and 3 need to reshape
  {
    _p.const_as_gamma->rank(1);
    _p.const_as_gamma->dim(0).set(_p.const_as_gamma->size<loco::DataType::FLOAT32>());
    _p.const_as_beta->rank(1);
    _p.const_as_beta->dim(0).set(_p.const_as_beta->size<loco::DataType::FLOAT32>());

    _p.const_as_gamma->shape_status(luci::ShapeStatus::UNDEFINED);
    _p.const_as_beta->shape_status(luci::ShapeStatus::UNDEFINED);
  }
}

luci::CircleInstanceNorm *FuseInstanceNorm::create_inst_norm(loco::Graph *graph)
{
  // Make Instance Norm to replace
  auto instance_norm = graph->nodes()->create<luci::CircleInstanceNorm>();
  instance_norm->input(_p.ifm);
  instance_norm->gamma(_p.const_as_gamma);
  instance_norm->beta(_p.const_as_beta);
  float epsilon = _p.const_as_epsilon->at<loco::DataType::FLOAT32>(0);
  instance_norm->epsilon(epsilon);
  if (_p.add_as_terminal != nullptr)
  {
    instance_norm->fusedActivationFunction(_p.add_as_terminal->fusedActivationFunction());
    // NOTE unique name should be assigned in export
    instance_norm->name("FusedInstanceNorm/" + _p.add_as_terminal->name());
  }
  else
  {
    // VERSION_4
    assert(_p.div != nullptr);
    instance_norm->fusedActivationFunction(_p.div->fusedActivationFunction());
    instance_norm->name("FusedInstanceNorm/" + _p.div->name());
  }

  return instance_norm;
}

template <> void FuseInstanceNorm::apply<InstanceNormPattern::PatternVersion::Version_1>()
{
  auto graph = _p.add_as_terminal->graph();

  reshape_gamma_beta();

  auto instance_norm = create_inst_norm(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p.mean_of_ifm),
    luci::get_origin(_p.sqdiff),
    luci::get_origin(_p.mean_as_variance),
    luci::get_origin(_p.add_as_variance),
    luci::get_origin(_p.rsqrt),
    luci::get_origin(_p.mul_gamma),
    luci::get_origin(_p.mul_as_scaled_ifm),
    luci::get_origin(_p.mul_as_scaled_mean),
    luci::get_origin(_p.sub),
    luci::get_origin(_p.add_as_terminal)};

  luci::add_origin(instance_norm, luci::composite_origin(origin_vec));

  replace(_p.add_as_terminal).with(instance_norm);
}

template <> void FuseInstanceNorm::apply<InstanceNormPattern::PatternVersion::Version_2>()
{
  auto graph = _p.add_as_terminal->graph();

  auto instance_norm = create_inst_norm(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p.mean_of_ifm),
    luci::get_origin(_p.sqdiff),
    luci::get_origin(_p.mean_as_variance),
    luci::get_origin(_p.add_as_variance),
    luci::get_origin(_p.pow),
    luci::get_origin(_p.sub),
    luci::get_origin(_p.div),
    luci::get_origin(_p.mul_gamma),
    luci::get_origin(_p.add_as_terminal)};

  luci::add_origin(instance_norm, luci::composite_origin(origin_vec));

  replace(_p.add_as_terminal).with(instance_norm);
}

template <> void FuseInstanceNorm::apply<InstanceNormPattern::PatternVersion::Version_3>()
{
  auto graph = _p.add_as_terminal->graph();

  reshape_gamma_beta();

  auto instance_norm = create_inst_norm(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p.mean_of_ifm),
    luci::get_origin(_p.sub),
    luci::get_origin(_p.mean_of_ifm_2),
    luci::get_origin(_p.sub_2),
    luci::get_origin(_p.square),
    luci::get_origin(_p.mean_as_variance),
    luci::get_origin(_p.sqrt),
    luci::get_origin(_p.add_as_variance),
    luci::get_origin(_p.div),
    luci::get_origin(_p.mul_gamma),
    luci::get_origin(_p.add_as_terminal)};

  luci::add_origin(instance_norm, luci::composite_origin(origin_vec));

  replace(_p.add_as_terminal).with(instance_norm);
}

template <> void FuseInstanceNorm::apply<InstanceNormPattern::PatternVersion::Version_4>()
{
  auto graph = _p.div->graph();

  auto instance_norm = create_inst_norm(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p.mean_of_ifm),
    luci::get_origin(_p.sub),
    luci::get_origin(_p.mean_of_ifm_2),
    luci::get_origin(_p.sub_2),
    luci::get_origin(_p.square),
    luci::get_origin(_p.mean_as_variance),
    luci::get_origin(_p.sqrt),
    luci::get_origin(_p.add_as_variance),
    luci::get_origin(_p.div)};

  luci::add_origin(instance_norm, luci::composite_origin(origin_vec));

  replace(_p.div).with(instance_norm);
}

void FuseInstanceNorm::apply()
{
  assert(_p.matched());

  switch (_p.version())
  {
    case InstanceNormPattern::PatternVersion::Version_1:
      apply<InstanceNormPattern::PatternVersion::Version_1>();
      break;
    case InstanceNormPattern::PatternVersion::Version_2:
      apply<InstanceNormPattern::PatternVersion::Version_2>();
      break;
    case InstanceNormPattern::PatternVersion::Version_3:
      apply<InstanceNormPattern::PatternVersion::Version_3>();
      break;
    case InstanceNormPattern::PatternVersion::Version_4:
      apply<InstanceNormPattern::PatternVersion::Version_4>();
      break;

    default:
      break;
  }
}

} // namespace

namespace
{

class PostFusion final
{
public:
  PostFusion(luci::CircleInstanceNorm *inst_norm) : _inst_norm(inst_norm) {}

public:
  bool process(void);

private:
  luci::CircleInstanceNorm *_inst_norm = nullptr;
};

bool PostFusion::process(void)
{
  bool changed = false;

  return changed;
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

bool fuse_instance_norm(luci::CircleAdd *add)
{
  InstanceNormPattern::PatternVersion pv = InstanceNormPattern::PatternVersion::Version_1;

  if (is_add_input_mul_const(add))
    pv = InstanceNormPattern::PatternVersion::Version_2;

  InstanceNormPattern pattern(add, pv);
  if (pattern.matched())
  {
    FuseInstanceNorm fuse(pattern);
    fuse.apply();
    return true;
  }

  if (pv == InstanceNormPattern::PatternVersion::Version_2)
  {
    // if Version_2 failed, try with Version_3
    pv = InstanceNormPattern::PatternVersion::Version_3;
    InstanceNormPattern pattern(add, pv);
    if (pattern.matched())
    {
      FuseInstanceNorm fuse(pattern);
      fuse.apply();
      return true;
    }
  }

  return false;
}

bool fuse_instance_norm(luci::CircleDiv *div)
{
  InstanceNormPattern::PatternVersion pv = InstanceNormPattern::PatternVersion::Version_4;

  InstanceNormPattern pattern(div, pv);
  if (pattern.matched())
  {
    FuseInstanceNorm fuse(pattern);
    fuse.apply();
    return true;
  }

  return false;
}

bool post_fusion(luci::CircleInstanceNorm *inst_norm)
{
  PostFusion postfusion(inst_norm);

  return postfusion.process();
}

} // namespace

namespace luci
{

bool FuseInstanceNormPass::run(loco::Graph *g)
{
  bool changed = false;

  // Check Version_1, Version_2, Version_3
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (not add)
      continue;

    if (fuse_instance_norm(add))
      changed = true;
  }

  // Check Version_4(from DIV) if MUL-ADD pattern is not found
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto div = dynamic_cast<luci::CircleDiv *>(node);
    if (not div)
      continue;

    if (fuse_instance_norm(div))
      changed = true;
  }

  // Post processing of FuseInstanceNorm
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto inst_norm = dynamic_cast<luci::CircleInstanceNorm *>(node);
    if (not inst_norm)
      continue;

    if (post_fusion(inst_norm))
      changed = true;
  }

  return changed;
}

} // namespace luci
