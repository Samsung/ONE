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

#include "luci/Pass/FuseRoPEPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cassert>
#include <set>

// Helper to fuse RoPE(Rotary Position Embedding)
namespace
{

/**
 * Below diagram shows RoPE pattern to fuse.
 * - this pattern will be replaced with one RoPE
 *
 * Mode: GPT-NEOX
 *                           [In]
 *                            |
 *                            V
 *               +----------- ifm ----------------+
 *               |             |                  |
 *               |             |                  |
 *               |             V                  |
 *               |       strided_slice_neg        V
 *               |             |             strided_slice
 *               |             V                  |
 *               |            neg                 |
 *               |             |                  |
 *               |             V                  |
 *               |       concatenation <----------+
 *               |             |
 *               |             |    sin_table(node)
 *               |             |        |
 *               |             V        |
 *   cos_table   |          mul_sin <---+
 *    (node)     |             |
 *      |        V             |
 *      + --- mul_cos          |
 *               |             |
 *               +---add_ofm <-+
 *                      |
 *                      V
 *                    [Out]
 */
class RoPEPattern final
{
public:
  enum PatternMode
  {
    Mode_Unknown,
    Mode_GPT_NEOX,
    Mode_GPT_J,
  };

  RoPEPattern(luci::CircleAdd *candidate, PatternMode pm)
  {
    assert(candidate);
    add_ofm = candidate;
    _pm = pm;
  }

private:
  template <enum PatternMode> bool match();

public:
  bool matched();
  bool matched() const { return _matched; }
  PatternMode mode() const { return _pm; }

public:
  // Context
  loco::Node *ifm = nullptr;
  loco::Node *sin_table = nullptr;
  loco::Node *cos_table = nullptr;
  luci::CircleTranspose *transpose = nullptr;
  luci::CircleTranspose *transpose_neg = nullptr;
  luci::CircleTranspose *transpose_mul = nullptr;
  luci::CircleStridedSlice *strided_slice_neg = nullptr;
  luci::CircleStridedSlice *strided_slice = nullptr;
  luci::CircleNeg *neg = nullptr;
  luci::CircleConcatenation *concat = nullptr;
  luci::CircleMul *mul_cos = nullptr;
  luci::CircleMul *mul_sin = nullptr;
  luci::CircleAdd *add_ofm = nullptr;

private:
  bool _matched = false;
  PatternMode _pm;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

template <> bool RoPEPattern::match<RoPEPattern::PatternMode::Mode_GPT_NEOX>()
{
  CHECK_OR_FALSE(luci::fill(&mul_cos, &mul_sin).with_commutative_args_of(add_ofm));
  CHECK_OR_FALSE(luci::fill(&ifm, &cos_table).with_commutative_args_of(mul_cos));

  auto ifm_circle = loco::must_cast<luci::CircleNode *>(ifm);
  CHECK_OR_FALSE(ifm_circle->shape_status() == luci::ShapeStatus::VALID);
  CHECK_OR_FALSE(ifm_circle->rank() == 4);
  CHECK_OR_FALSE(ifm_circle->dim(3).known());

  CHECK_OR_FALSE(luci::fill(&concat, &sin_table).with_commutative_args_of(mul_sin));
  CHECK_OR_FALSE(concat->numValues() == 2);

  strided_slice = dynamic_cast<luci::CircleStridedSlice *>(concat->values(1));
  CHECK_OR_FALSE(strided_slice);

  neg = dynamic_cast<luci::CircleNeg *>(concat->values(0));
  CHECK_OR_FALSE(neg);

  strided_slice_neg = dynamic_cast<luci::CircleStridedSlice *>(neg->x());
  CHECK_OR_FALSE(strided_slice_neg);

  _matched = true;
  return true;
}

bool RoPEPattern::matched()
{
  if (_matched)
    return true;

  switch (_pm)
  {
    case PatternMode::Mode_GPT_NEOX:
      return match<PatternMode::Mode_GPT_NEOX>();

    default:
      break;
  }

  throw std::runtime_error("Invalid RoPE Pattern.");
}

#undef CHECK_OR_FALSE

class FuseRoPE final
{
public:
  FuseRoPE(const RoPEPattern &p) : _p(p) {}

public:
  void apply(void);

private:
  template <RoPEPattern::PatternMode> void apply(void);

private:
  luci::CircleRoPE *create_rope(loco::Graph *graph);

private:
  const RoPEPattern &_p;
};

luci::CircleRoPE *FuseRoPE::create_rope(loco::Graph *graph)
{
  assert(graph);

  auto rope = graph->nodes()->create<luci::CircleRoPE>();
  rope->input(_p.ifm);
  rope->cos_table(_p.cos_table);
  rope->sin_table(_p.sin_table);

  if (_p.mode() == RoPEPattern::PatternMode::Mode_GPT_NEOX)
    rope->mode(luci::RoPEMode::GPT_NEOX);

  rope->name(_p.add_ofm->name() + "_rope");
  return rope;
}

template <> void FuseRoPE::apply<RoPEPattern::PatternMode::Mode_GPT_NEOX>()
{
  auto graph = _p.add_ofm->graph();

  auto rope = create_rope(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p.strided_slice), luci::get_origin(_p.strided_slice_neg),
    luci::get_origin(_p.neg),           luci::get_origin(_p.concat),
    luci::get_origin(_p.mul_cos),       luci::get_origin(_p.mul_sin),
    luci::get_origin(_p.add_ofm)};

  luci::add_origin(rope, luci::composite_origin(origin_vec));

  rope->cos_table(_p.cos_table);
  rope->sin_table(_p.sin_table);

  replace(_p.add_ofm).with(rope);
}

void FuseRoPE::apply()
{
  assert(_p.matched());

  switch (_p.mode())
  {
    case RoPEPattern::PatternMode::Mode_GPT_NEOX:
      apply<RoPEPattern::PatternMode::Mode_GPT_NEOX>();
      break;

    default:
      break;
  }
}
} // namespace

namespace
{

bool fuse_rope(luci::CircleAdd *add)
{
  RoPEPattern::PatternMode pm = RoPEPattern::PatternMode::Mode_GPT_NEOX;

  // TODO support other modes (ex. GPT_J)

  RoPEPattern pattern(add, pm);
  if (pattern.matched())
  {
    FuseRoPE fuse(pattern);
    fuse.apply();
    return true;
  }
  return false;
}

} // namespace

namespace luci
{

bool FuseRoPEPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto add = dynamic_cast<luci::CircleAdd *>(node);
    if (not add)
      continue;

    if (fuse_rope(add))
      changed = true;
  }

  return changed;
}

} // namespace luci
