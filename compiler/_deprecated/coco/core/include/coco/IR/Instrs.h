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

#ifndef __COCO_IR_INSTRS_H__
#define __COCO_IR_INSTRS_H__

#include "coco/IR/Instr.h"

#include "coco/IR/ElemID.h"

#include "coco/IR/Bag.h"
#include "coco/IR/Object.h"

#include "coco/IR/Def.h"
#include "coco/IR/Use.h"
#include "coco/IR/Read.h"
#include "coco/IR/Update.h"

#include "coco/IR/Step.h"

#include <map>

namespace coco
{

/**
 * @brief Evaluate an Object from a given Op
 */
class Eval final : public Instr, public Object::Producer
{
public:
  explicit Eval();

public:
  Eval *asEval(void) override { return this; }
  const Eval *asEval(void) const override { return this; }

public:
  Instr *loc(void) override { return this; }

public:
  Object *out(void) const { return _out.value(); }
  void out(Object *obj) { _out.value(obj); }

public:
  Op *op(void) const { return _step.op(); }
  void op(Op *op) { _step.op(op); }

private:
  Def _out;
  Step _step;
};

/**
 * @brief Index-wise element transfer between two objects
 *
 * Given two objects "src" and "dst" of the same kind/shape, "copy(src, dst)"
 * denotes index-wise element transfer.
 *
 * For example, the following pseudo-code describes "copy(src, dat)"
 * when both src and dst are a feature map of the shape B x C x H x W:
 *
 * for each valid index b, ch, row, col:
 *   load the "src->at(b, ch, row, col)"-th element from bag(src)
 *   store it as the "dst->at(b, ch, row, col)"-th element of bag(dst)
 *
 * In principle, "copy" is unnecessary as it is always possible to rewrite "copy"
 * as a "shuffle" below. However, "shuffle"-based optimization is too heavy as it
 * requires much of iterations.
 */
class Copy final : public Instr, public Object::Producer, public Object::Consumer
{
public:
  Copy() : _from{this}, _into{this}
  {
    // DO NOTHING
  }

public:
  Copy *asCopy(void) override { return this; }
  const Copy *asCopy(void) const override { return this; }

public:
  Instr *loc(void) override { return this; }

public:
  Object *from(void) const { return _from.value(); }
  void from(Object *o) { _from.value(o); }

public:
  Object *into(void) const { return _into.value(); }
  void into(Object *o) { _into.value(o); }

private:
  Use _from;
  Def _into;
};

/**
 * @brief Generic element transfer
 */
class Shuffle final : public Instr, public Bag::Reader, public Bag::Updater
{
public:
  Shuffle() : _from{this}, _into{this}
  {
    // DO NOTHING
  }

public:
  Shuffle *asShuffle(void) override { return this; }
  const Shuffle *asShuffle(void) const override { return this; }

public:
  Instr *loc(void) override { return this; }

public:
  Bag *from(void) const { return _from.bag(); }
  void from(Bag *bag);

public:
  Bag *into(void) const { return _into.bag(); }
  void into(Bag *);

public:
  /**
   * @brief Return the number of Element-wise transfers
   *
   * NOTE size() SHOULD BE identical to range().size()
   */
  uint32_t size(void) const;

  /// @brief Return a set of elements in the destination bag that Shuffle will update
  std::set<ElemID> range(void) const;

public:
  /// @brief Return true if a given elem is updated after execution
  bool defined(const ElemID &dst) const { return _content.find(dst) != _content.end(); }

public:
  /**
   * Let M be the return of at(N). This means that N-th element in the destination
   * bag will be filled with the value of M-th element in the source bag.
   *
   * NOTE at(n) may be undefined on partial shuffle
   */
  const ElemID &at(const ElemID &dst) const { return _content.at(dst); }

public:
  void insert(const ElemID &from, const ElemID &into);

private:
  Read _from;
  Update _into;

private:
  std::map<ElemID /* DST */, ElemID /* SRC */> _content;
};

} // namespace coco

#endif // __COCO_IR_INSTRS_H__
