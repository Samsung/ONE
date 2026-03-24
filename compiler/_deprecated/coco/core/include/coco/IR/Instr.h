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

#ifndef __COCO_IR_INSTR_H__
#define __COCO_IR_INSTR_H__

#include "coco/IR/Bag.h"
#include "coco/IR/Block.forward.h"
#include "coco/IR/Instr.forward.h"
#include "coco/IR/InstrIndex.h"
#include "coco/IR/Entity.h"

#include "coco/ADT/DLinkedList.h"

#include <cassert>
#include <stdexcept>

namespace coco
{

#define INSTR(Name) class Name;
#include "coco/IR/Instr.lst"
#undef INSTR

using InstrList = coco::DLinkedList<Instr, Block>::Head;

/**
 * @brief Base interface on explicit computation steps in coco IR
 *
 * NOTE Input/output is explicit in Instr, but implicit in Op
 * NOTE Instr is may (not always) be a combination of multiple NN operations
 *
 * One may find a set of supported instructions from "Instrs.h"
 *
 * >> How to add a new base instruction in coco IR <<
 *
 * To introduce a new instruction (whose name is INS),
 *   1. Append "INSTR(INS)" to "Instr.lst"
 *   2. Declare class INS which inherits Instr class in "Instrs.h"
 *      NOTE This class SHOULD be default constructible
 *
 */
class Instr : public coco::DLinkedList<Instr, Block>::Node, public Entity
{
public:
  friend void DLinkedList<Instr, Block>::joined(Block *, Instr *);
  friend void DLinkedList<Instr, Block>::leaving(Block *, Instr *);

public:
  Instr() = default;

public:
  Instr(const Instr &) = delete;
  Instr(Instr &&) = delete;

public:
  virtual ~Instr()
  {
    if (parent())
    {
      // NOTE It is safe to invoke detach here (although "Instr" is not a final class)
      //      as "leaving" hook accesses only the internal of "Instr" class
      detach();
    }
  }

public:
#define INSTR(Name)                                \
  virtual Name *as##Name(void) { return nullptr; } \
  virtual const Name *as##Name(void) const { return nullptr; }
#include "coco/IR/Instr.lst"
#undef INSTR

public:
  /**
   * @brief Instr visitor interface
   *
   * WARN Use this interface only for coco-internal classes
   *      (to minimize changes upon Instr extension)
   */
  template <typename T> struct IVisitor
  {
    virtual ~IVisitor() = default;

#define INSTR(Name) virtual T visit(const Name *) = 0;
#include "coco/IR/Instr.lst"
#undef INSTR
  };

  template <typename T> struct Visitor : public IVisitor<T>
  {
    virtual ~Visitor() = default;

#define INSTR(Name) \
  T visit(const Name *) override { throw std::runtime_error{"NYI"}; }
#include "coco/IR/Instr.lst"
#undef INSTR
  };

public:
  template <typename T> T accept(IVisitor<T> *v) const
  {
#define INSTR(Name)          \
  if (auto ins = as##Name()) \
  {                          \
    return v->visit(ins);    \
  }
#include "coco/IR/Instr.lst"
#undef INSTR
    throw std::runtime_error{"unreachable"};
  }

  template <typename T> T accept(IVisitor<T> &v) const { return accept(&v); }
  template <typename T> T accept(IVisitor<T> &&v) const { return accept(&v); }

public:
  const InstrIndex &index(void) const { return _index; }

private:
  InstrIndex _index;
};

/**
 * @brief Return true if a given instruction is of T type
 *
 * @note "ins" cannot be a null pointer
 */
template <typename T> bool isa(const Instr *ins)
{
  assert(ins != nullptr);
  return dynamic_cast<const T *>(ins) != nullptr;
}

/**
 * @brief Cast as a derived instruction
 *
 * @note "safe_cast<T>(ins)" returns a null pointer if "ins" is not of T type
 * @note "safe_cast<T>(ins)" returns a null pointer if "ins" is a null pointer
 */
template <typename T> T *safe_cast(Instr *ins)
{
  // NOTE dynamic_cast<T *>(nullptr) returns nullptr
  return dynamic_cast<T *>(ins);
}

} // namespace coco

#endif // __COCO_IR_INSTR_H__
