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

#include "luci/Pass/FuseGeluPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cmath>

#include <cassert>

// Helper to fuse Gelu
namespace
{

// Float comparison
bool same(float a, float b) { return fabs(a - b) < 1e-5; }

class GeluPatternBase
{
public:
  GeluPatternBase(luci::CircleMul *candidate) { _pattern_last_node = candidate; }

  virtual ~GeluPatternBase() = default;

public:
  virtual bool matched() = 0;

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleMul *_mul_sqrt = nullptr;
  luci::CircleCustom *_erf = nullptr;
  luci::CircleCustomOut *_erf_out = nullptr;
  luci::CircleAdd *_add_one = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleMul *_mul_half = nullptr;
  luci::CircleConst *_const_sqrt = nullptr;
  luci::CircleConst *_const_one = nullptr;
  luci::CircleConst *_const_half = nullptr;
  luci::CircleMul *_pattern_last_node = nullptr;
};

/**
 * Below diagram shows Gelu pattern to fuse.
 * - Gelu(x) = 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
 * - the below pattern will be replaced with one Gelu
 *
 *           [In]
 *            |
 *            V
 *     +---- ifm
 *     |      |
 *     |      V
 *     |  mul_sqrt (1/sqrt(2) = 0.707106..)
 *     |      |
 *     |      V
 *     |     erf
 *     |      |
 *     |      V
 *     |   add_one (1.0)
 *     |      |
 *     |      V
 *     +---> mul
 *            |
 *            V
 *         mul_half (0.5)
 *            |
 *            V
 *          [Out]
 *
 */
class GeluPattern1 final : public GeluPatternBase
{
public:
  GeluPattern1(luci::CircleMul *candidate) : GeluPatternBase(candidate)
  {
    assert(candidate);
    _mul_half = candidate;
  }

public:
  bool matched() override;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool GeluPattern1::matched()
{
  // check pattern
  CHECK_OR_FALSE(luci::fill(&_mul, &_const_half).with_commutative_args_of(_mul_half));
  CHECK_OR_FALSE(luci::fill(&_ifm, &_add_one).with_commutative_args_of(_mul));
  CHECK_OR_FALSE(luci::fill(&_erf_out, &_const_one).with_commutative_args_of(_add_one));

  if (auto erf = dynamic_cast<luci::CircleCustom *>(_erf_out->input()))
    _erf = erf;

  CHECK_OR_FALSE(_erf != nullptr);

  // Check erf
  CHECK_OR_FALSE(_erf->custom_code() == "Erf");
  CHECK_OR_FALSE(_erf->numInputs() == 1);
  CHECK_OR_FALSE(_erf->numOutputs() == 1);

  if (auto mul_sqrt = dynamic_cast<luci::CircleMul *>(_erf->inputs(0)))
    _mul_sqrt = mul_sqrt;

  CHECK_OR_FALSE(_mul_sqrt != nullptr);

  CHECK_OR_FALSE(luci::fill(&_ifm, &_const_sqrt).with_commutative_args_of(_mul_sqrt));

  CHECK_OR_FALSE(_mul_sqrt->x() == _ifm);
  CHECK_OR_FALSE(_mul->x() == _ifm);

  // Check Activation to be NONE
  CHECK_OR_FALSE(_mul_sqrt->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_add_one->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_mul->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_mul_half->fusedActivationFunction() == luci::FusedActFunc::NONE);

  // check _const_sqrt condition
  CHECK_OR_FALSE(_const_sqrt->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_const_sqrt->size<loco::DataType::FLOAT32>() == 1);
  CHECK_OR_FALSE(::same(_const_sqrt->at<loco::DataType::FLOAT32>(0), sqrtf(0.5f)));

  // check if _const_half is 0.5 (fp32)
  CHECK_OR_FALSE(_const_half->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_const_half->size<loco::DataType::FLOAT32>() == 1);
  CHECK_OR_FALSE(_const_half->at<loco::DataType::FLOAT32>(0) == 0.5);

  // check _const_one condition
  CHECK_OR_FALSE(_const_one->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_const_one->size<loco::DataType::FLOAT32>() == 1);
  CHECK_OR_FALSE(_const_one->at<loco::DataType::FLOAT32>(0) == 1);

  return true;
}

#undef CHECK_OR_FALSE

class FuseGelu final
{
public:
  FuseGelu(const GeluPatternBase *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleGelu *create_gelu(loco::Graph *graph);

private:
  const GeluPatternBase *_p;
};

luci::CircleGelu *FuseGelu::create_gelu(loco::Graph *graph)
{
  assert(graph);

  auto gelu = graph->nodes()->create<luci::CircleGelu>();
  gelu->features(_p->_ifm);
  // TODO Support approximate = True pattern
  gelu->approximate(false);
  gelu->name(_p->_mul_half->name() + "_gelu");
  return gelu;
}

void FuseGelu::apply()
{
  auto graph = _p->_mul_half->graph();

  auto gelu = create_gelu(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_mul_sqrt), luci::get_origin(_p->_erf), luci::get_origin(_p->_add_one),
    luci::get_origin(_p->_mul), luci::get_origin(_p->_mul_half)};

  luci::add_origin(gelu, luci::composite_origin(origin_vec));

  replace(_p->_mul_half).with(gelu);
}

} // namespace

namespace
{

bool fuse_gelu(luci::CircleMul *mul)
{
  assert(mul);

  // check first pattern
  GeluPattern1 pattern(mul);
  if (pattern.matched())
  {
    FuseGelu fuse(&pattern);
    fuse.apply();
    return true;
  }
  return false;
}

} // namespace

namespace luci
{

bool FuseGeluPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (not mul)
      continue;

    if (fuse_gelu(mul))
      changed = true;
  }

  return changed;
}

} // namespace luci
