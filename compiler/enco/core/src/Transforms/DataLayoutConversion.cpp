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

#include "DataLayoutConversion.h"
#include "Session.h"
#include "IRUtils.h"

#include "coex/IR.h"

#include <coco/IR/FeatureLayouts.h>
#include <coco/IR/KernelLayouts.h>

#include <nncc/core/ADT/feature/Layout.h>
#include <nncc/core/ADT/kernel/Layout.h>

#include <nncc/core/ADT/feature/HWCLayout.h>
#include <nncc/core/ADT/kernel/NHWCLayout.h>

#include <set>

namespace
{

coco::Copy *make_copy(coco::FeatureObject *from, coco::FeatureObject *into)
{
  auto m = from->module();
  assert(m != nullptr);
  assert(from->module() == into->module());

  auto copy = m->entity()->instr()->create<coco::Copy>();

  copy->from(from);
  copy->into(into);

  return copy;
}

coco::FeatureObject *clone_feature(const coco::FeatureObject *oldobj)
{
  auto module = oldobj->module();
  auto newobj = module->entity()->object()->create<coco::FeatureObject>();
  newobj->layout(coco::FeatureLayouts::BHWC::create(oldobj->shape()));

  if (oldobj->bag() != nullptr)
  {
    using nncc::core::ADT::feature::num_elements;

    // NOTE The size of bag should be at least "BxHxWxC" as "newobj" uses BHWC layout
    const uint32_t batch = newobj->layout()->batch();
    const uint32_t count = num_elements(newobj->layout()->shape());
    const uint32_t bag_size = batch * count;

    // Clone bag only when there is a backing bag for a given feature object
    auto newbag = module->entity()->bag()->create(bag_size);
    newobj->bag(newbag);
  }

  return newobj;
}

/**
 * @brief Insert Copy before Load if necessary
 *
 * @require "load" should be bounded
 */
void insert_copy_before_load(coco::Load *load)
{
  assert(load->parent() != nullptr);
  assert(load->parent()->parent() != nullptr);

  if (auto obj = load->object())
  {
    if (auto ifm = obj->asFeature())
    {
      if (ifm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
      {
        auto oldobj = ifm;
        auto newobj = clone_feature(oldobj);

        load->object(newobj);

        auto copy = make_copy(oldobj, newobj);
        copy->insertBefore(load->parent());
      }
    }
  }
}

/**
 * @brief Insert Copy after Eval if necessary
 */
void insert_copy_after_eval(coco::Eval *eval)
{
  if (auto out = eval->out())
  {
    if (auto ofm = out->asFeature())
    {
      if (ofm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
      {
        auto oldobj = ofm;
        auto newobj = clone_feature(oldobj);

        eval->out(newobj);

        auto copy = make_copy(newobj, oldobj);
        copy->insertAfter(eval);
      }
    }
  }
}

/**
 * @brief Insert copy (for data layout change) before/after ANNDepthConcatF if necessary
 */
void convert_data_layout(ANNDepthConcatF *concat)
{
  if (auto out = concat->out())
  {
    if (auto ofm = out->asFeature())
    {
      if (ofm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
      {
        auto oldobj = ofm;
        auto newobj = clone_feature(oldobj);

        concat->out(newobj);

        auto copy = make_copy(newobj, oldobj);
        copy->insertAfter(concat);
      }
    }
  }

  if (auto obj = concat->fst())
  {
    if (auto ifm = obj->asFeature())
    {
      if (ifm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
      {
        auto oldobj = ifm;
        auto newobj = clone_feature(oldobj);

        concat->fst(newobj);

        auto copy = make_copy(oldobj, newobj);
        copy->insertBefore(concat);
      }
    }
  }

  if (auto obj = concat->snd())
  {
    if (auto ifm = obj->asFeature())
    {
      if (ifm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
      {
        auto oldobj = ifm;
        auto newobj = clone_feature(oldobj);

        concat->snd(newobj);

        auto copy = make_copy(oldobj, newobj);
        copy->insertBefore(concat);
      }
    }
  }
}

/**
 * @brief Update convolution kernel data layout
 */
void change_conv2d_kernel_layout(coco::Conv2D *conv)
{
  auto m = conv->module();
  assert(m != nullptr);
  auto d = enco::data(enco::session(m));
  assert(d != nullptr);

  auto old_obj = conv->ker();
  assert(old_obj != nullptr);
  auto old_bag = old_obj->bag();
  assert(old_bag != nullptr);

  if (old_obj->layout()->id() == coco::KernelLayouts::NHWC::uid())
  {
    // Skip if kernel already uses NHWC layout
    return;
  }

  const auto &ker_shape = old_obj->shape();

  assert(d->allocated(old_bag));

  auto new_bag = m->entity()->bag()->create(old_bag->size());
  auto new_obj = m->entity()->object()->create<coco::KernelObject>();

  new_obj->bag(new_bag);
  new_obj->layout(coco::KernelLayouts::NHWC::create(ker_shape));

  d->f32()->allocate(new_bag);

  auto src = d->f32()->read(old_obj);
  auto dst = d->f32()->access(new_obj);

  const auto ker_N = ker_shape.count();
  const auto ker_C = ker_shape.depth();
  const auto ker_H = ker_shape.height();
  const auto ker_W = ker_shape.width();

  for (uint32_t n = 0; n < ker_N; ++n)
  {
    for (uint32_t ch = 0; ch < ker_C; ++ch)
    {
      for (uint32_t row = 0; row < ker_H; ++row)
      {
        for (uint32_t col = 0; col < ker_W; ++col)
        {
          dst->at(n, ch, row, col) = src->at(n, ch, row, col);
        }
      }
    }
  }

  conv->ker(new_obj);
  d->release(old_bag);
}

} // namespace

namespace
{

/**
 * @brief Return the set of all of bounded Load Op(s) in a given module
 *
 * @note  'bounded' means it will be exectuted
 */
std::set<coco::Load *> loads(coco::Module *m)
{
  std::set<coco::Load *> res;

  for (uint32_t n = 0; n < m->entity()->op()->size(); ++n)
  {
    auto op = m->entity()->op()->at(n);

    // Skip if this op is dangling
    if (op->parent() == nullptr)
    {
      continue;
    }

    // Skip if eval instruction of this op is dangling
    if (op->parent()->parent() == nullptr)
    {
      continue;
    }

    if (auto load = m->entity()->op()->at(n)->asLoad())
    {
      res.insert(load);
    }
  }

  return res;
}

/**
 * @brief Return the set of every (allocated) Eval instruction in a given module
 */
std::set<coco::Eval *> evals(coco::Module *m)
{
  std::set<coco::Eval *> res;

  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    if (auto eval = m->entity()->instr()->at(n)->asEval())
    {
      res.insert(eval);
    }
  }

  return res;
}

/**
 * @brief Return the set of allocated Conv2D op in a given module
 */
std::set<coco::Conv2D *> convs(coco::Module *m)
{
  std::set<coco::Conv2D *> res;

  for (uint32_t n = 0; n < m->entity()->op()->size(); ++n)
  {
    if (auto op = m->entity()->op()->at(n)->asConv2D())
    {
      res.insert(op);
    }
  }

  return res;
}

/**
 * @brief Return the set of "bounded" ANNDepthConcatF instructions
 */
std::set<ANNDepthConcatF *> depth_concats(coco::Module *m)
{
  std::set<ANNDepthConcatF *> res;

  for (auto ins : enco::instr_sequence(m))
  {
    if (auto depth_concat_f = coco::safe_cast<ANNDepthConcatF>(ins))
    {
      res.insert(depth_concat_f);
    }
  }

  return res;
}

class NormalizePass
{
private:
  void runOnModule(coco::Module *m) const;

public:
  void runOnCode(enco::Code *) const;
};

void NormalizePass::runOnModule(coco::Module *m) const
{
  // Insert Copy before all Load Op (if necessary)
  for (auto load : loads(m))
  {
    insert_copy_before_load(load);
  }

  // Insert Copy after all Eval Instr (if necessary)
  for (auto eval : evals(m))
  {
    insert_copy_after_eval(eval);
  }

  // Change Kernel Layout of Conv2D opertion (if necessary)
  for (auto conv : convs(m))
  {
    change_conv2d_kernel_layout(conv);
  }

  // Insert Copy (for Layout Conversion) before/after ANNDepthConcatF instructions (if necessary)
  for (auto depth_concat : depth_concats(m))
  {
    convert_data_layout(depth_concat);
  }
}

void NormalizePass::runOnCode(enco::Code *code) const { runOnModule(code->module()); }

} // namespace

namespace enco
{

void convert_data_layout(enco::Code *code)
{
  NormalizePass pass;
  pass.runOnCode(code);
}

} // namespace enco
