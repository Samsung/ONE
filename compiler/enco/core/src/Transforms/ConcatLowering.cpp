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

#include "CopyLowering.h"
#include "IRUtils.h"

#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <set>
#include <cassert>

using namespace nncc::core::ADT;

namespace
{

inline uint32_t as_tensor_axis(const coco::ConcatF::Axis &axis)
{
  switch (axis)
  {
    case coco::ConcatF::Axis::Batch:
      return 0;
    case coco::ConcatF::Axis::Depth:
      return 1;
    case coco::ConcatF::Axis::Height:
      return 2;
    case coco::ConcatF::Axis::Width:
      return 3;
    default:
      break;
  };

  throw std::invalid_argument{"axis is unknown value"};
}

tensor::Shape as_tensor_shape(const coco::FeatureLayout *l)
{
  assert(l != nullptr);

  tensor::Shape res;

  res.resize(4);

  res.dim(as_tensor_axis(coco::ConcatF::Axis::Batch)) = l->batch();
  res.dim(as_tensor_axis(coco::ConcatF::Axis::Depth)) = l->depth();
  res.dim(as_tensor_axis(coco::ConcatF::Axis::Height)) = l->height();
  res.dim(as_tensor_axis(coco::ConcatF::Axis::Width)) = l->width();

  return res;
}

coco::ElemID as_element_index(const coco::FeatureLayout *l, const tensor::Index &idx)
{
  assert(l != nullptr);
  assert(idx.rank() == 4);

  const auto b = idx.at(as_tensor_axis(coco::ConcatF::Axis::Batch));
  const auto ch = idx.at(as_tensor_axis(coco::ConcatF::Axis::Depth));
  const auto row = idx.at(as_tensor_axis(coco::ConcatF::Axis::Height));
  const auto col = idx.at(as_tensor_axis(coco::ConcatF::Axis::Width));

  return l->at(b, ch, row, col);
}

std::set<coco::Eval *> candidates(coco::Module *m)
{
  std::set<coco::Eval *> res;

  for (auto ins : enco::instr_sequence(m))
  {
    if (auto eval = ins->asEval())
    {
      if (eval->op()->asConcatF())
      {
        res.insert(eval);
      }
    }
  }

  return res;
}

} // namespace

namespace enco
{

void lower_concat(enco::Code *code)
{
  auto m = code->module();

  for (auto eval : candidates(m))
  {
    auto concat_f = eval->op()->asConcatF();
    assert(concat_f != nullptr);

    auto left_feature = concat_f->left()->asLoad()->object()->asFeature();
    assert(left_feature != nullptr);
    auto left_shape = as_tensor_shape(left_feature->layout());

    auto right_feature = concat_f->right()->asLoad()->object()->asFeature();
    assert(right_feature != nullptr);
    auto right_shape = as_tensor_shape(right_feature->layout());

    auto out_feature = eval->out()->asFeature();
    assert(out_feature != nullptr);
    auto out_shape = as_tensor_shape(out_feature->layout());

    auto concat_axe = as_tensor_axis(concat_f->axis());

    // Lower: Left -> Output
    {
      auto src_feature = left_feature;
      auto src_shape = left_shape;

      auto ins = m->entity()->instr()->create<coco::Shuffle>();

      assert(src_feature->bag() != nullptr);
      assert(out_feature->bag() != nullptr);

      ins->from(src_feature->bag());
      ins->into(out_feature->bag());

      for (tensor::IndexEnumerator e{src_shape}; e.valid(); e.advance())
      {
        tensor::Index src_index = e.current();
        tensor::Index out_index = e.current();

        auto from = as_element_index(src_feature->layout(), src_index);
        auto into = as_element_index(out_feature->layout(), out_index);

        ins->insert(from, into);
      }

      ins->insertAfter(eval);
    }

    // Lower: Right -> Output
    {
      auto src_feature = right_feature;
      auto src_shape = right_shape;

      auto ins = m->entity()->instr()->create<coco::Shuffle>();

      assert(src_feature->bag() != nullptr);
      assert(out_feature->bag() != nullptr);

      ins->from(src_feature->bag());
      ins->into(out_feature->bag());

      for (tensor::IndexEnumerator e{src_shape}; e.valid(); e.advance())
      {
        tensor::Index src_index = e.current();
        tensor::Index out_index = e.current();

        out_index.at(concat_axe) = out_index.at(concat_axe) + left_shape.dim(concat_axe);

        auto from = as_element_index(src_feature->layout(), src_index);
        auto into = as_element_index(out_feature->layout(), out_index);

        ins->insert(from, into);
      }

      ins->insertAfter(eval);
    }

    // Unlink "Eval" and "ConcatF" op tree
    eval->op(nullptr);

    // Delete "Concat" op tree
    m->entity()->op()->destroy(concat_f->left());
    m->entity()->op()->destroy(concat_f->right());
    m->entity()->op()->destroy(concat_f);

    // Deatch "Eval" instruction from the block
    eval->detach();

    // Delete "Eval" instruction
    m->entity()->instr()->destroy(eval);
  }
}

} // namespace enco
