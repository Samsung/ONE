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

#include "ConstantFolding.h"
#include "Session.h"

#include <queue>
#include <cmath>
#include <cassert>

namespace
{

/**
 * @brief is_constant_bag(b) returns true if the bag "b" has corresponding weight
 */
bool is_constant_bag(coco::Bag *b)
{
  auto m = b->module();
  auto d = enco::data(m);
  return d->allocated(b);
}

class ConstantBagEnumerator
{
public:
  ConstantBagEnumerator(enco::Code *code) : _code{code}
  {
    // DO NOTHING
  }

public:
  template <typename Callable> void enumerate(Callable cb) const
  {
    auto m = _code->module();

    for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
    {
      auto b = m->entity()->bag()->at(n);

      if (is_constant_bag(b))
      {
        cb(b);
      }
    }
  }

private:
  enco::Code *_code;
};

template <typename Callable> void operator<<(const ConstantBagEnumerator &e, Callable &&cb)
{
  e.enumerate(std::forward<Callable>(cb));
}

ConstantBagEnumerator constant_bag_enumerator(enco::Code *code)
{
  return ConstantBagEnumerator{code};
}

} // namespace

namespace
{

/**
 * @brief Take the first element from the queue
 * @note The queue SHOULD have at least one element.
 */
template <typename T> T take(std::queue<T> &q)
{
  assert(q.size() > 0);
  auto res = q.front();
  q.pop();
  return res;
}

} // namespace

namespace
{

void fold_constant(std::queue<coco::Bag *> &q, coco::Copy *copy)
{
  auto m = copy->module();
  auto d = enco::data(m);

  auto src_obj = copy->from();
  auto src_bag = src_obj->bag();

  auto dst_obj = copy->into();
  auto dst_bag = dst_obj->bag();

  // Output calculation should not be folded
  // TODO Reduce code duplication of this kind
  if (dst_bag->isOutput())
  {
    return;
  }

  // NOTE d->allocated(bag) returns true if bag has corresponding initial
  //      values (e.g. convolution kernel)
  assert(d->allocated(src_bag));
  assert(!d->allocated(dst_bag));

  // TODO Support other data type
  auto src_span = d->f32()->weight(src_bag);

  assert(src_span.data() != nullptr);

  auto src_feature = src_obj->asFeature();
  auto dst_feature = dst_obj->asFeature();

  // TODO Support other object type
  if (src_feature == nullptr || dst_feature == nullptr)
  {
    return;
  }

  assert(src_feature != nullptr);
  assert(dst_feature != nullptr);

  // Allocate weight for destination
  d->f32()->allocate(dst_bag);

  auto dst_span = d->f32()->weight(dst_bag);

  assert(src_feature->layout()->batch() == dst_feature->layout()->batch());
  assert(src_feature->layout()->depth() == dst_feature->layout()->depth());
  assert(src_feature->layout()->height() == dst_feature->layout()->height());
  assert(src_feature->layout()->width() == dst_feature->layout()->width());

  uint32_t const B = src_feature->layout()->batch();
  uint32_t const C = src_feature->layout()->depth();
  uint32_t const H = src_feature->layout()->height();
  uint32_t const W = src_feature->layout()->width();

  for (uint32_t b = 0; b < B; ++b)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          auto src_ind = src_feature->layout()->at(b, ch, row, col);
          auto dst_ind = dst_feature->layout()->at(b, ch, row, col);

          dst_span[dst_ind.value()] = src_span[src_ind.value()];
        }
      }
    }
  }

  // Let's detach copy
  copy->from(nullptr);
  copy->into(nullptr);
  copy->detach();

  // Let's visit destination bag!
  q.push(dst_bag);
}

template <typename Callable>
void fold_constant_op(std::queue<coco::Bag *> &q, coco::UnaryOp *op, Callable evaluate)
{
  auto m = op->module();
  auto d = enco::data(m);

  auto ins = op->parent();
  auto eval = ins->asEval();

  // UnaryOp has only one arg
  auto src_obj = *(op->uses().begin());
  auto src_bag = src_obj->bag();

  auto dst_obj = eval->out();
  auto dst_bag = dst_obj->bag();

  // Output calculation should not be folded
  // TODO Reduce code duplication of this kind
  if (dst_bag->isOutput())
  {
    return;
  }

  assert(d->allocated(src_bag));
  assert(!d->allocated(dst_bag));

  // TODO Support other data type
  auto src_span = d->f32()->weight(src_bag);
  assert(src_span.data() != nullptr);

  auto src_feature = src_obj->asFeature();
  auto dst_feature = dst_obj->asFeature();

  // TODO Support other object type
  if (src_feature == nullptr || dst_feature == nullptr)
  {
    return;
  }

  assert(src_feature != nullptr);
  assert(dst_feature != nullptr);

  // Allocate weight for destination
  d->f32()->allocate(dst_bag);
  auto dst_span = d->f32()->weight(dst_bag);

  assert(src_feature->layout()->batch() == dst_feature->layout()->batch());
  assert(src_feature->layout()->depth() == dst_feature->layout()->depth());
  assert(src_feature->layout()->height() == dst_feature->layout()->height());
  assert(src_feature->layout()->width() == dst_feature->layout()->width());

  uint32_t const B = src_feature->layout()->batch();
  uint32_t const C = src_feature->layout()->depth();
  uint32_t const H = src_feature->layout()->height();
  uint32_t const W = src_feature->layout()->width();

  for (uint32_t b = 0; b < B; ++b)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          auto src_ind = src_feature->layout()->at(b, ch, row, col);
          auto dst_ind = dst_feature->layout()->at(b, ch, row, col);

          evaluate(&dst_span[dst_ind.value()], src_span[src_ind.value()]);
        }
      }
    }
  }

  // Let's detach eval
  eval->out(nullptr);
  eval->detach();

  // Let's visit destination bag!
  q.push(dst_bag);
}

template <typename Callable>
void fold_constant_op(std::queue<coco::Bag *> &q, coco::BinaryOp *op, Callable evaluate)
{
  auto m = op->module();
  auto d = enco::data(m);

  auto ins = op->parent();
  auto eval = ins->asEval();

  // Already folded by the other bag
  if (!eval->out())
  {
    return;
  }

  auto lhs_load = op->left()->asLoad();
  auto lhs_obj = lhs_load->object();
  auto lhs_bag = lhs_obj->bag();

  auto rhs_load = op->right()->asLoad();
  auto rhs_obj = rhs_load->object();
  auto rhs_bag = rhs_obj->bag();

  auto dst_obj = eval->out();
  auto dst_bag = dst_obj->bag();

  // Output calculation should not be folded
  // TODO Reduce code duplication of this kind
  if (dst_bag->isOutput())
  {
    return;
  }

  // The other bag is non-constant
  if (!d->allocated(lhs_bag) || !d->allocated(rhs_bag))
  {
    return;
  }

  assert(d->allocated(lhs_bag));
  assert(d->allocated(rhs_bag));
  assert(!d->allocated(dst_bag));

  // TODO Support other data type
  auto lhs_span = d->f32()->weight(lhs_bag);
  auto rhs_span = d->f32()->weight(rhs_bag);
  assert(lhs_span.data() != nullptr);
  assert(rhs_span.data() != nullptr);

  auto lhs_feature = lhs_obj->asFeature();
  auto rhs_feature = rhs_obj->asFeature();
  auto dst_feature = dst_obj->asFeature();

  // TODO Support other object type
  if (lhs_feature == nullptr || rhs_feature == nullptr || dst_feature == nullptr)
  {
    return;
  }

  assert(lhs_feature != nullptr);
  assert(rhs_feature != nullptr);
  assert(dst_feature != nullptr);

  // Allocate weight for destination
  d->f32()->allocate(dst_bag);
  auto dst_span = d->f32()->weight(dst_bag);

  assert(lhs_feature->layout()->batch() == rhs_feature->layout()->batch());
  assert(lhs_feature->layout()->depth() == rhs_feature->layout()->depth());
  assert(lhs_feature->layout()->height() == rhs_feature->layout()->height());
  assert(lhs_feature->layout()->width() == rhs_feature->layout()->width());

  assert(lhs_feature->layout()->batch() == dst_feature->layout()->batch());
  assert(lhs_feature->layout()->depth() == dst_feature->layout()->depth());
  assert(lhs_feature->layout()->height() == dst_feature->layout()->height());
  assert(lhs_feature->layout()->width() == dst_feature->layout()->width());

  uint32_t const B = lhs_feature->layout()->batch();
  uint32_t const C = lhs_feature->layout()->depth();
  uint32_t const H = lhs_feature->layout()->height();
  uint32_t const W = lhs_feature->layout()->width();

  for (uint32_t b = 0; b < B; ++b)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          auto lhs_ind = lhs_feature->layout()->at(b, ch, row, col);
          auto rhs_ind = rhs_feature->layout()->at(b, ch, row, col);
          auto dst_ind = dst_feature->layout()->at(b, ch, row, col);

          evaluate(&dst_span[dst_ind.value()], lhs_span[lhs_ind.value()],
                   rhs_span[rhs_ind.value()]);
        }
      }
    }
  }

  // Let's detach eval
  eval->out(nullptr);
  eval->detach();

  // Let's visit destination bag!
  q.push(dst_bag);
}

void fold_constant(std::queue<coco::Bag *> &q, coco::Eval *eval)
{
  // TODO Support other data types
  if (auto op = eval->op()->asSqrt())
  {
    fold_constant_op(q, op, [](float *dst, float value) { *dst = std::sqrt(value); });
  }
  else if (auto op = eval->op()->asAdd())
  {
    fold_constant_op(q, op, [](float *dst, float lhs, float rhs) { *dst = lhs + rhs; });
  }
  else if (auto op = eval->op()->asSub())
  {
    fold_constant_op(q, op, [](float *dst, float lhs, float rhs) { *dst = lhs - rhs; });
  }
  else if (auto op = eval->op()->asMul())
  {
    fold_constant_op(q, op, [](float *dst, float lhs, float rhs) { *dst = lhs * rhs; });
  }
  else if (auto op = eval->op()->asDiv())
  {
    fold_constant_op(q, op, [](float *dst, float lhs, float rhs) { *dst = lhs / rhs; });
  }
  else
  {
    // Not supported opteration, do nothing
    // TODO Support other operations
  }
}

void fold_constant(std::queue<coco::Bag *> &q, coco::Instr *ins)
{
  if (auto copy = coco::safe_cast<coco::Copy>(ins))
  {
    fold_constant(q, copy);
    return;
  }
  if (auto eval = coco::safe_cast<coco::Eval>(ins))
  {
    fold_constant(q, eval);
    return;
  }

  // TODO Add more cases for constant folding
}

} // namespace

namespace enco
{

void fold_constants(enco::Code *code)
{
  std::queue<coco::Bag *> q;

  // Collect the initial set of "constant" bag
  constant_bag_enumerator(code) << [&q](coco::Bag *bag) { q.push(bag); };

  while (!q.empty())
  {
    auto candidate_bag = take(q);

    // Scan the readers of each candidate bag
    for (auto reader : coco::readers(candidate_bag))
    {
      // TODO Decide how to handle the reader with unknown instruction
      if (auto ins = reader->loc())
      {
        fold_constant(q, ins);
      }
    }
  }
}

} // namespace enco
