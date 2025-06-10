/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseRmsNormPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cmath>
#include <cassert>

namespace
{

/**
 * Below diagram shows RMS normalization pattern to fuse.
 * - this pattern will be replaced with one RmsNorm
 *
 *           [In]
 *            |
 *            V
 *     +---- ifm ----+
 *     |      |      |
 *     |      V      |
 *     |     mul <---+
 *     |      |
 *     |      V
 *     |     mean
 *     |      |
 *     |      V
 *     |     add_epsilon
 *     |      |
 *     |      V
 *     |     rsqrt
 *     |      |
 *     |      V
 *     +---> mul_input
 *            |
 *            V
 *          [Out]
 */

class RmsNormPattern final
{
public:
  RmsNormPattern(luci::CircleMul *candidate)
  {
    assert(candidate); // FIX_CALLER_UNLESS
    _mul_input = candidate;
  }

public:
  bool matched();

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleMul *_mul_pow = nullptr;
  luci::CircleMean *_mean = nullptr;
  luci::CircleAdd *_add_epsilon = nullptr;
  luci::CircleRsqrt *_rsqrt = nullptr;
  luci::CircleMul *_mul_input = nullptr;
  luci::CircleConst *_const_epsilon = nullptr;
  luci::CircleConst *_const_gamma = nullptr;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

luci::CircleConst *make_const_one(loco::Graph *graph, float value)
{
  auto const_one = graph->nodes()->create<luci::CircleConst>();
  const_one->dtype(loco::DataType::FLOAT32);
  const_one->rank(1);
  const_one->dim(0) = 1;
  const_one->shape_status(luci::ShapeStatus::VALID);
  const_one->size<loco::DataType::FLOAT32>(1);
  const_one->at<loco::DataType::FLOAT32>(0) = value;
  return const_one;
}

bool RmsNormPattern::matched()
{
  CHECK_OR_FALSE(luci::fill(&_ifm, &_rsqrt).with_commutative_args_of(_mul_input));
  _add_epsilon = dynamic_cast<luci::CircleAdd *>(_rsqrt->x());
  CHECK_OR_FALSE(_add_epsilon);
  CHECK_OR_FALSE(luci::fill(&_mean, &_const_epsilon).with_commutative_args_of(_add_epsilon));
  CHECK_OR_FALSE(_const_epsilon->dtype() == loco::DataType::FLOAT32);
  _mul_pow = dynamic_cast<luci::CircleMul *>(_mean->input());
  CHECK_OR_FALSE(_mul_pow);
  CHECK_OR_FALSE(_mul_pow->x() == _ifm);
  CHECK_OR_FALSE(_mul_pow->y() == _ifm);

  assert(_const_gamma == nullptr);

  /*
   NOTE: Current FuseRmsNormPass assumes no gamma(scale).
   But, RmsNorm kernel expects gamma.
   So, it creates default gamma(1.0).
  */
  auto graph = _mul_input->graph();
  _const_gamma = make_const_one(graph, 1.0f);
  _const_gamma->name(_mul_input->name() + "/gamma");

  return true;
}
#undef CHECK_OR_FALSE

class FuseRmsNorm final
{
public:
  FuseRmsNorm(const RmsNormPattern *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleRmsNorm *create_rms_norm(loco::Graph *graph);

private:
  const RmsNormPattern *_p = nullptr;
};

luci::CircleRmsNorm *FuseRmsNorm::create_rms_norm(loco::Graph *graph)
{
  assert(graph);

  auto rms_norm = graph->nodes()->create<luci::CircleRmsNorm>();
  rms_norm->input(_p->_ifm);
  rms_norm->gamma(_p->_const_gamma);
  float epsilon = _p->_const_epsilon->at<loco::DataType::FLOAT32>(0);
  rms_norm->epsilon(epsilon);

  rms_norm->name("FusedRmsNorm/" + _p->_mul_input->name());

  return rms_norm;
}

void FuseRmsNorm::apply()
{
  auto graph = _p->_mul_input->graph();

  auto rms_norm = create_rms_norm(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_mul_pow),     luci::get_origin(_p->_mean),
    luci::get_origin(_p->_add_epsilon), luci::get_origin(_p->_rsqrt),
    luci::get_origin(_p->_mul_input),
  };

  luci::add_origin(rms_norm, luci::composite_origin(origin_vec));

  replace(_p->_mul_input).with(rms_norm);
}

} // namespace

namespace
{

bool fuse_rms_norm(luci::CircleMul *mul)
{
  assert(mul);

  RmsNormPattern pattern(mul);
  if (pattern.matched())
  {
    FuseRmsNorm fuse(&pattern);
    fuse.apply();
    return true;
  }

  return false;
}

} // namespace

namespace luci
{

bool FuseRmsNormPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (not mul)
      continue;

    if (fuse_rms_norm(mul))
      changed = true;
  }
  return changed;
}

} // namespace luci
