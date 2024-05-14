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

#include "luci/Pass/FuseRsqrtPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <cmath>
#include <cassert>

namespace
{

// Float comparison
bool same(float a, float b) { return fabs(a - b) < 1e-5; }

class RsqrtPattern
{
public:
  RsqrtPattern(luci::CircleDiv *candidate)
  {
    assert(candidate); // FIX_CALLER_UNLESS
    _div = candidate;
  }

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

public:
  bool matched()
  {
    // Check pattern
    CHECK_OR_FALSE(luci::fill(&_div_const, &_sqrt).with_args_of(_div));
    _ifm = loco::must_cast<luci::CircleNode *>(_sqrt->x());

    CHECK_OR_FALSE(_div->fusedActivationFunction() == luci::FusedActFunc::NONE);

    // Check div_const = 1
    switch (_div->dtype())
    {
      case loco::DataType::S16:
        CHECK_OR_FALSE(_div_const->quantparam() != nullptr);
        CHECK_OR_FALSE(_div_const->quantparam()->scale.size() == 1);
        CHECK_OR_FALSE(_div_const->quantparam()->zerop.size() == 1);
        CHECK_OR_FALSE(_div_const->quantparam()->zerop.at(0) == 0);
        CHECK_OR_FALSE(_div_const->size<loco::DataType::S16>() == 1);
        CHECK_OR_FALSE(same(1.0, _div_const->at<loco::DataType::S16>(0) *
                                   _div_const->quantparam()->scale.at(0)));
        break;
      // TODO Support more dtypes
      default:
        return false;
    }

    return true;
  }
#undef CHECK_OR_FALSE

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleSqrt *_sqrt = nullptr;
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_div_const = nullptr;
};

class FuseRsqrt final
{
public:
  FuseRsqrt(const RsqrtPattern *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleRsqrt *create_rsqrt(loco::Graph *graph);

private:
  const RsqrtPattern *_p;
};

luci::CircleRsqrt *FuseRsqrt::create_rsqrt(loco::Graph *graph)
{
  assert(graph);

  auto rsqrt = graph->nodes()->create<luci::CircleRsqrt>();
  rsqrt->x(_p->_ifm);
  rsqrt->name(_p->_div->name() + "_rsqrt");

  luci::copy_quantparam(_p->_div, rsqrt);

  return rsqrt;
}

void FuseRsqrt::apply()
{
  auto graph = _p->_div->graph();

  auto rsqrt = create_rsqrt(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_sqrt), luci::get_origin(_p->_div), luci::get_origin(_p->_div_const)};

  luci::add_origin(rsqrt, luci::composite_origin(origin_vec));

  replace(_p->_div).with(rsqrt);
}

} // namespace

namespace
{

bool fuse_rsqrt(luci::CircleDiv *div)
{
  assert(div);

  RsqrtPattern pattern(div);
  if (pattern.matched())
  {
    FuseRsqrt fuse(&pattern);
    fuse.apply();
    return true;
  }

  return false;
}

} // namespace

namespace luci
{

bool FuseRsqrtPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto div = dynamic_cast<luci::CircleDiv *>(node);
    if (not div)
      continue;

    if (fuse_rsqrt(div))
      changed = true;
  }

  return changed;
}

} // namespace luci
