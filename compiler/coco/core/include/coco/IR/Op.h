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

/**
 * @file  Op.h
 * @brief This header file declares "Op" class and several traits related with "Op"
 */
#ifndef __COCO_IR_OP_H__
#define __COCO_IR_OP_H__

#include "coco/IR/Object.forward.h"
#include "coco/IR/Instr.forward.h"
#include "coco/IR/Step.forward.h"
#include "coco/IR/Part.h"
#include "coco/IR/Entity.h"

#include <cstdint>
#include <set>

#include <stdexcept>

namespace coco
{

#define OP(Name) class Name;
#include "coco/IR/Op.lst"
#undef OP

/**
 * @brief Base interface on all supported NN operations
 */
struct Op : public Entity
{
  friend class Step;
  friend class Part;

  virtual ~Op();

  /**
   * @brief Return the number of arguments (# of child Ops)
   */
  virtual uint32_t arity(void) const = 0;

  /**
   * @brief Return N-th argument
   *
   * @note The behavior of arg(n) is defined only when n < artiy()
   */
  virtual Op *arg(uint32_t n) const = 0;

  /**
   * @brief Return a set of object(s) used during execution
   *
   * NOTE There is no 'def' method as Op is not allowed to define a new object
   */
  virtual std::set<Object *> uses(void) const = 0;

#define OP(Name)                                   \
  virtual Name *as##Name(void) { return nullptr; } \
  virtual const Name *as##Name(void) const { return nullptr; }
#include "coco/IR/Op.lst"
#undef OP

  /**
   * @brief Op visitor interface
   *
   * WARN Use this interface only for coco-internal classes
   *      (to minimize changes upon Op extension)
   */
  template <typename T> struct IVisitor
  {
    virtual ~IVisitor() = default;

#define OP(Name) virtual T visit(const Name *) = 0;
#include "coco/IR/Op.lst"
#undef OP
  };

  template <typename T> struct Visitor : public IVisitor<T>
  {
    virtual ~Visitor() = default;

#define OP(Name) \
  T visit(const Name *) override { throw std::runtime_error{"NYI"}; }
#include "coco/IR/Op.lst"
#undef OP
  };

  template <typename T> T accept(IVisitor<T> *v) const
  {
#define OP(Name)            \
  if (auto op = as##Name()) \
  {                         \
    return v->visit(op);    \
  }
#include "coco/IR/Op.lst"
#undef OP
    throw std::runtime_error{"unreachable"};
  }

  template <typename T> T accept(IVisitor<T> &v) const { return accept(&v); }
  template <typename T> T accept(IVisitor<T> &&v) const { return accept(&v); }

public:
  /**
   * @brief Op mutator interface
   *
   * WARN Use this interface only for coco-internal classes
   *      (to minimize changes upon Instr extension)
   */
  struct IMutator
  {
    virtual ~IMutator() = default;

#define OP(Name) virtual void mutate(Name *) = 0;
#include "coco/IR/Op.lst"
#undef OP
  };

  struct Mutator : public IMutator
  {
    virtual ~Mutator() = default;

#define OP(Name) \
  void mutate(Name *) override { throw std::runtime_error{"NYI"}; }
#include "coco/IR/Op.lst"
#undef OP
  };

  void accept(IMutator *m)
  {
#define OP(Name)            \
  if (auto op = as##Name()) \
  {                         \
    return m->mutate(op);   \
  }
#include "coco/IR/Op.lst"
#undef OP
    throw std::runtime_error{"unreachable"};
  }

  void accept(IMutator &m) { return accept(&m); }
  void accept(IMutator &&m) { return accept(&m); }

public:
  Instr *parent(void) const;

  /// @brief Return a pointer to the parent Op
  Op *up(void) const;

private:
  /**
   * @brief A link to Instr from Op
   *
   * WARN Update this field only through Step
   */
  Step *_step = nullptr;

  /**
   * @brief A link to a parent Op
   *
   * WARN Update this field only through Part
   * NOTE An "Op" CANNOT have a link to a parent Op if it is linked to an "Instr"
   */
  Part *_part = nullptr;
};

/**
 * @brief Op with a single argument
 */
class UnaryOp : public Op
{
public:
  explicit UnaryOp();

public:
  UnaryOp(const UnaryOp &) = delete;
  UnaryOp(UnaryOp &&) = delete;

public:
  virtual ~UnaryOp() = default;

public:
  uint32_t arity(void) const final;
  Op *arg(uint32_t n) const final;

  std::set<Object *> uses(void) const final;

public:
  Op *arg(void) const { return _arg.child(); }
  void arg(Op *arg) { _arg.child(arg); }

private:
  /// @brief Link to Op's argument
  Part _arg;
};

/**
 * @brief Op with two arguments
 */
class BinaryOp : public Op
{
public:
  explicit BinaryOp();

public:
  BinaryOp(const BinaryOp &) = delete;
  BinaryOp(BinaryOp &&) = delete;

public:
  virtual ~BinaryOp() = default;

public:
  uint32_t arity(void) const final;
  Op *arg(uint32_t n) const final;

  std::set<Object *> uses(void) const final;

public:
  Op *left(void) const { return _left.child(); }
  void left(Op *op) { _left.child(op); }

public:
  Op *right(void) const { return _right.child(); }
  void right(Op *op) { _right.child(op); }

private:
  /// @brief Left-hand side (LHS) argument
  Part _left;
  /// @brief Right-hand side (RHS) argument
  Part _right;
};

/**
 * @brief Return the root Op from a given Op node
 *
 * @note root(op) == op holds for a root op
 */
Op *root(Op *);

} // namespace coco

#endif // __COCO_IR_OP_H__
