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

#include "luci/Pass/ExtractGeluFromOptFCPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

#include <cmath>

#include <cassert>

// Helper to fuse Gelu
namespace
{

// Float comparison
bool same(float a, float b) { return fabs(a - b) < 1e-5; }

luci::CircleConst *multiply_const(luci::CircleConst *node, float multiplier)
{
  auto cloned = luci::clone(node);

  assert(node->dtype() == loco::DataType::FLOAT32);   // FIX_CALLER_UNLESS
  assert(cloned->dtype() == loco::DataType::FLOAT32); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < cloned->size<loco::DataType::FLOAT32>(); i++)
  {
    cloned->at<loco::DataType::FLOAT32>(i) *= multiplier;
  }

  luci::add_origin(cloned, luci::get_origin(node));

  return cloned;
}

/**
 * Below diagram shows the target pattern.
 * - The pattern will be converted to FC (front) -> Gelu -> FC (back).
 * - FC (front) has the same weights with fc1
 * - FC (back)'s weights is twice of fc3's weights
 *
 *     +---- [In]
 *     |      |
 *     |      V
 *     |     fc2 (w = w of fc1 * sqrt(0.5)) -> const folded
 *     |      |
 *    fc1     V
 *     |     erf
 *     |      |
 *     |      V
 *     |   add_one (1.0)
 *     |      |
 *     |      V
 *     +---> mul
 *            |
 *            V
 *           fc3
 *            |
 *            V
 *          [Out]
 *
 */
class FCGeluFCPattern final
{
public:
  FCGeluFCPattern(luci::CircleFullyConnected *cand)
  {
    assert(cand);
    _fc3 = cand;
  }

public:
  bool matched();

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleFullyConnected *_fc1 = nullptr;
  luci::CircleFullyConnected *_fc2 = nullptr;
  luci::CircleFullyConnected *_fc3 = nullptr;
  luci::CircleCustom *_erf = nullptr;
  luci::CircleCustomOut *_erf_out = nullptr;
  luci::CircleAdd *_add_one = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_const_one = nullptr;
  luci::CircleConst *_fc1_w = nullptr;
  luci::CircleConst *_fc2_w = nullptr;
  luci::CircleConst *_fc3_w = nullptr;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool FCGeluFCPattern::matched()
{
  // check pattern
  _fc3_w = dynamic_cast<luci::CircleConst *>(_fc3->weights());
  CHECK_OR_FALSE(_fc3_w != nullptr);

  _mul = dynamic_cast<luci::CircleMul *>(_fc3->input());
  CHECK_OR_FALSE(_mul != nullptr);

  CHECK_OR_FALSE(luci::fill(&_fc1, &_add_one).with_commutative_args_of(_mul));

  _fc1_w = dynamic_cast<luci::CircleConst *>(_fc1->weights());
  CHECK_OR_FALSE(_fc1_w != nullptr);

  CHECK_OR_FALSE(_fc1->weights_format() == luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  _ifm = loco::must_cast<luci::CircleNode *>(_fc1->input());

  CHECK_OR_FALSE(luci::fill(&_erf_out, &_const_one).with_commutative_args_of(_add_one));

  _erf = dynamic_cast<luci::CircleCustom *>(_erf_out->input());
  CHECK_OR_FALSE(_erf != nullptr);

  // Check erf
  CHECK_OR_FALSE(_erf->custom_code() == "Erf");
  CHECK_OR_FALSE(_erf->numInputs() == 1);
  CHECK_OR_FALSE(_erf->numOutputs() == 1);

  _fc2 = dynamic_cast<luci::CircleFullyConnected *>(_erf->inputs(0));
  CHECK_OR_FALSE(_fc2 != nullptr);
  _fc2_w = dynamic_cast<luci::CircleConst *>(_fc2->weights());
  CHECK_OR_FALSE(_fc2_w != nullptr);

  CHECK_OR_FALSE(_fc2->weights_format() == luci::CircleFullyConnected::WeightsFormat::DEFAULT);
  CHECK_OR_FALSE(_ifm == _fc2->input());

  // Check Activation to be NONE
  CHECK_OR_FALSE(_mul->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_add_one->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_fc1->fusedActivationFunction() == luci::FusedActFunc::NONE);
  CHECK_OR_FALSE(_fc2->fusedActivationFunction() == luci::FusedActFunc::NONE);
  // fc3 can have activation

  // Check dtype
  CHECK_OR_FALSE(_fc1->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_fc2->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_fc3->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_erf->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_erf_out->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_add_one->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_mul->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_fc1_w->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_fc2_w->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_fc3_w->dtype() == loco::DataType::FLOAT32);

  // Check all fc layers have no bias
  // TODO Remove this restriction if necessary
  CHECK_OR_FALSE(dynamic_cast<luci::CircleOutputExclude *>(_fc1->bias()) != nullptr);
  CHECK_OR_FALSE(dynamic_cast<luci::CircleOutputExclude *>(_fc2->bias()) != nullptr);
  CHECK_OR_FALSE(dynamic_cast<luci::CircleOutputExclude *>(_fc3->bias()) != nullptr);

  // Check _const_one condition
  CHECK_OR_FALSE(_const_one->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(_const_one->size<loco::DataType::FLOAT32>() == 1);
  CHECK_OR_FALSE(_const_one->at<loco::DataType::FLOAT32>(0) == 1);

  // Check fc2_w = fc1_w * sqrt(0.5)
  CHECK_OR_FALSE(_fc1_w->size<loco::DataType::FLOAT32>() ==
                 _fc2_w->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < _fc1_w->size<loco::DataType::FLOAT32>(); i++)
  {
    const auto fc1_val = _fc1_w->at<loco::DataType::FLOAT32>(i);
    const auto fc2_val = _fc2_w->at<loco::DataType::FLOAT32>(i);
    CHECK_OR_FALSE(::same(fc1_val * sqrtf(0.5f), fc2_val));
  }

  return true;
}

#undef CHECK_OR_FALSE

class ExtractGeluFromOptFC final
{
public:
  ExtractGeluFromOptFC(const FCGeluFCPattern *p) : _p(p) {}

public:
  void apply(void);

private:
  // Create FC -> Gelu -> FC pattern and set front/back
  void create_fc_gelu_fc(luci::CircleFullyConnected *&front, luci::CircleFullyConnected *&back);

private:
  const FCGeluFCPattern *_p;
};

void ExtractGeluFromOptFC::create_fc_gelu_fc(luci::CircleFullyConnected *&front,
                                             luci::CircleFullyConnected *&back)
{
  auto graph = _p->_fc1->graph();
  assert(graph);

  front = loco::must_cast<luci::CircleFullyConnected *>(luci::clone_node(_p->_fc1, graph));
  front->weights(_p->_fc1->weights());
  front->bias(_p->_fc1->bias());
  luci::add_origin(front, luci::get_origin(_p->_fc1));

  auto gelu = graph->nodes()->create<luci::CircleGelu>();
  gelu->features(front);
  // TODO Support approximate = True pattern
  gelu->approximate(false);
  gelu->name(_p->_erf->name() + "_gelu");
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_fc2), luci::get_origin(_p->_erf), luci::get_origin(_p->_add_one),
    luci::get_origin(_p->_mul)};
  luci::add_origin(gelu, luci::composite_origin(origin_vec));

  back = loco::must_cast<luci::CircleFullyConnected *>(luci::clone_node(_p->_fc3, graph));
  back->input(gelu);
  back->weights(multiply_const(_p->_fc3_w, 2.0f /* multiplier */));
  back->bias(_p->_fc3->bias());
  luci::add_origin(back, luci::get_origin(_p->_fc3));
}

void ExtractGeluFromOptFC::apply()
{
  luci::CircleFullyConnected *front = nullptr;
  luci::CircleFullyConnected *back = nullptr;
  create_fc_gelu_fc(front, back);

  assert(front); // FIX_ME_UNLESS
  assert(back);  // FIX_ME_UNLESS

  front->input(_p->_ifm);

  replace(_p->_fc3).with(back);
}

} // namespace

namespace
{

bool extract_gelu(luci::CircleFullyConnected *fc)
{
  assert(fc);

  FCGeluFCPattern pattern(fc);
  if (pattern.matched())
  {
    ExtractGeluFromOptFC extract(&pattern);
    extract.apply();
    return true;
  }

  return false;
}

} // namespace

namespace luci
{

bool ExtractGeluFromOptFCPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (extract_gelu(fc))
      changed = true;
  }

  return changed;
}

} // namespace luci
