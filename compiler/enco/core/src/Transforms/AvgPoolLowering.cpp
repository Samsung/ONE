/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "AvgPoolLowering.h"
#include "IRUtils.h"

#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/feature/Shape.h>
#include <nncc/core/ADT/feature/HWCLayout.h>

#include <set>
#include <cassert>

using namespace nncc::core::ADT;
using nncc::core::ADT::feature::num_elements;

namespace
{

bool empty(coco::Padding2D *pad)
{
  return (pad->top() == 0) && (pad->bottom() == 0) && (pad->left() == 0) && (pad->right() == 0);
}

/**
 * @brief Return a set of AvgPool2D operations (in Eval instruction) that SHOULD be lowered
 */
std::set<coco::AvgPool2D *> candidates(coco::Module *m)
{
  std::set<coco::AvgPool2D *> res;

  for (auto I : enco::instr_sequence(m))
  {
    if (auto eval = I->asEval())
    {
      if (auto avgpool = eval->op()->asAvgPool2D())
      {
        /* Originally it was preferred to use `auto load = avgpool->arg()->asLoad()' for
         * consitent style with other if statements.
         * Someone may think compiler will be happy because `load` in `if` statement can
         * be considered as a use, however, it turend out that it is not the case.
         */
        if (avgpool->arg()->asLoad())
        {
          if (avgpool->divisor() == coco::AvgPool2D::Divisor::Static)
          {
            res.insert(avgpool);
          }
        }
      }
    }
  }

  return res;
}

} // namespace

namespace
{
namespace ShapeTransform
{

class Pad
{
public:
  Pad(const coco::Padding2D *pad) : _pad{pad}
  {
    // DO NOTHING
  }

public:
  /// @brief Return the expected OFM shape for a given IFM shape
  feature::Shape forward(const feature::Shape &ifm_shape) const
  {
    const uint32_t OFM_C = ifm_shape.depth();
    const uint32_t OFM_H = ifm_shape.height() + _pad->top() + _pad->bottom();
    const uint32_t OFM_W = ifm_shape.width() + _pad->left() + _pad->right();

    return feature::Shape{OFM_C, OFM_H, OFM_W};
  }

private:
  const coco::Padding2D *_pad;
};

} // namespace ShapeTransform

ShapeTransform::Pad shape_xform(const coco::Padding2D *pad) { return ShapeTransform::Pad{pad}; }

} // namespace

namespace
{

class PadInstrBuilder final
{
public:
  PadInstrBuilder(const coco::Padding2D *pad) : _pad{pad}
  {
    // DO NOTHING
  }

public:
  coco::Instr *build(coco::FeatureObject *ifm_obj, coco::FeatureObject *ofm_obj) const
  {
    assert(ifm_obj->module() == ofm_obj->module());
    auto m = ifm_obj->module();
    assert(m != nullptr);

    auto load_op = m->entity()->op()->create<coco::Load>();

    load_op->object(ifm_obj);

    auto pad_op = m->entity()->op()->create<coco::PadF>();

    pad_op->arg(load_op);

    pad_op->pad()->top(_pad->top());
    pad_op->pad()->bottom(_pad->bottom());
    pad_op->pad()->left(_pad->left());
    pad_op->pad()->right(_pad->right());

    auto pad_instr = m->entity()->instr()->create<coco::Eval>();

    pad_instr->out(ofm_obj);
    pad_instr->op(pad_op);

    return pad_instr;
  }

private:
  const coco::Padding2D *_pad;
};

PadInstrBuilder pad_instr_builder(const coco::Padding2D *pad) { return PadInstrBuilder{pad}; }

} // namespace

namespace
{

class AvgPoolRewritePass
{
private:
  void runOnModule(coco::Module *m) const;

public:
  void runOnCode(enco::Code *) const;
};

void AvgPoolRewritePass::runOnModule(coco::Module *m) const
{
  // Lower AvgPool2D op that resides in Eval instruction
  for (auto avgpool : candidates(m))
  {
    auto ins = avgpool->parent();
    auto load = avgpool->arg()->asLoad();

    assert(ins != nullptr);
    assert(load != nullptr);
    assert(avgpool->divisor() == coco::AvgPool2D::Divisor::Static);

    if (empty(avgpool->pad()))
    {
      // NOTE If there is no padding, Static and PaddingExcluded schemes are equivalent
      avgpool->divisor(coco::AvgPool2D::Divisor::PaddingExcluded);
    }
    else
    {
      // Before: Static AvgPool2D with Padding
      // After: PadF; PaddingExcluded AvgPool2D without Padding

      // Create PadF
      auto ifm_obj = load->object()->asFeature();
      assert(ifm_obj != nullptr);

      auto pad_shape = shape_xform(avgpool->pad()).forward(ifm_obj->shape());
      auto pad_bag = m->entity()->bag()->create(num_elements(pad_shape));
      auto pad_obj = m->entity()->object()->create<coco::FeatureObject>();

      pad_obj->bag(pad_bag);
      pad_obj->layout(coco::FeatureLayouts::BHWC::create(pad_shape));

      auto pad_instr = pad_instr_builder(avgpool->pad()).build(ifm_obj, pad_obj);

      // Insert PadF before AvgPool2D
      pad_instr->insertBefore(ins);

      // Rewrite AvgPool2D as PaddingExcluded AvgPool2D without Padding
      load->object(pad_obj);

      avgpool->divisor(coco::AvgPool2D::Divisor::PaddingExcluded);
      avgpool->pad()->top(0);
      avgpool->pad()->bottom(0);
      avgpool->pad()->left(0);
      avgpool->pad()->right(0);
    }
  }
}

void AvgPoolRewritePass::runOnCode(enco::Code *code) const { runOnModule(code->module()); }

} // namespace

namespace enco
{

void lower_avgpool(enco::Code *code)
{
  AvgPoolRewritePass pass;
  pass.runOnCode(code);
}

} // namespace enco
