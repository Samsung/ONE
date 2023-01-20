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

#include "luci/Pass/FusePReluPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cassert>

// Helper to fuse PRelu
namespace
{

/**
 * Below diagram shows PRelu pattern to fuse.
 * - this pattern will be replaced with one PRelu
 *
 *           [In]
 *            |
 *            V
 *     +---- ifm ----+
 *     |      |      |
 *     |      |      V
 *     |      |     abs
 *     |      V      |
 *     |     sub <---+
 *     |      |
 *     |      V
 *     |   mul_alpha (alpha of PRelu)
 *     |      |
 *     V      V
 *    relu mul_half (0.5)
 *     |      |
 *     |      V
 *     +---> add
 *            |
 *            V
 *          [Out]
 *
 */
class PReluPattern final
{
public:
  PReluPattern(luci::CircleAdd *candidate)
  {
    assert(candidate);
    _add_ofm = candidate;
  }

public:
  bool matched();

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleRelu *_relu = nullptr;
  luci::CircleAbs *_abs = nullptr;
  luci::CircleSub *_sub = nullptr;
  luci::CircleMul *_mul_alpha = nullptr;
  luci::CircleMul *_mul_half = nullptr;
  luci::CircleAdd *_add_ofm = nullptr;
  luci::CircleConst *_const_alpha = nullptr;
  luci::CircleConst *_const_half = nullptr;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool PReluPattern::matched()
{
  // check pattern
  CHECK_OR_FALSE(luci::fill(&_relu, &_mul_half).with_commutative_args_of(_add_ofm));
  CHECK_OR_FALSE(luci::fill(&_mul_alpha, &_const_half).with_commutative_args_of(_mul_half));
  CHECK_OR_FALSE(luci::fill(&_sub, &_const_alpha).with_commutative_args_of(_mul_alpha));

  CHECK_OR_FALSE(luci::fill(&_ifm, &_abs).with_args_of(_sub));

  CHECK_OR_FALSE(_relu->features() == _ifm);
  CHECK_OR_FALSE(_abs->x() == _ifm);

  // Check Activation to be NONE
  CHECK_OR_FALSE(_sub->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_mul_alpha->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_mul_half->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_add_ofm->fusedActivationFunction() == luci::FusedActFunc::NONE);

  // TODO support other types?
  // check if _const_half is really FLOAT32 & 0.5
  CHECK_OR_FALSE(_const_half->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_const_half->size<loco::DataType::FLOAT32>() == 1);
  CHECK_OR_FALSE(_const_half->at<loco::DataType::FLOAT32>(0) == 0.5);

  // check _const_alpha condition
  CHECK_OR_FALSE(_const_alpha->dtype() == loco::DataType::FLOAT32);
  // TODO add more if needed

  return true;
}

#undef CHECK_OR_FALSE

class FusePRelu final
{
public:
  FusePRelu(const PReluPattern &p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CirclePRelu *create_prelu(loco::Graph *graph);

private:
  const PReluPattern &_p;
};

luci::CirclePRelu *FusePRelu::create_prelu(loco::Graph *graph)
{
  assert(graph);

  auto prelu = graph->nodes()->create<luci::CirclePRelu>();
  prelu->input(_p._ifm);
  prelu->alpha(_p._const_alpha);
  prelu->name(_p._add_ofm->name() + "_prelu");
  return prelu;
}

void FusePRelu::apply()
{
  auto graph = _p._add_ofm->graph();

  auto prelu = create_prelu(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p._relu),      luci::get_origin(_p._abs),      luci::get_origin(_p._sub),
    luci::get_origin(_p._mul_alpha), luci::get_origin(_p._mul_half), luci::get_origin(_p._add_ofm)};

  luci::add_origin(prelu, luci::composite_origin(origin_vec));

  replace(_p._add_ofm).with(prelu);
}

} // namespace

namespace
{

bool fuse_prelu(luci::CircleAdd *add)
{
  assert(add);

  PReluPattern pattern(add);
  if (pattern.matched())
  {
    FusePRelu fuse(pattern);
    fuse.apply();
    return true;
  }
  return false;
}

} // namespace

namespace luci
{

bool FusePReluPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (not add)
      continue;

    if (fuse_prelu(add))
      changed = true;
  }

  return changed;
}

} // namespace luci
