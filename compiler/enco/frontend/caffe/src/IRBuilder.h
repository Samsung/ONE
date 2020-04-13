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

#ifndef __IR_BUILDER_H__
#define __IR_BUILDER_H__

#include "coco/IR/Module.h"

#include <deque>

/**
 * coco IR builders
 */

class OpBuilder
{
public:
  OpBuilder(coco::Module *module) : _module{module}
  {
    // module SHOULD BE valid
    assert(_module != nullptr);
  }

public:
  /**
   * @brief Return true if the internal stack is empty
   */
  bool empty(void) const { return _stack.empty(); }

  /**
   * @brief Return the operation at the top of the internal stack
   */
  coco::Op *top(void) const
  {
    assert(_stack.size() > 0);
    return _stack.front();
  }

  /**
   * @brief Push op onto the internal stack
   *
   * BEFORE| Stack
   * AFTER | Op; Stack
   */
  OpBuilder &push(coco::Op *op)
  {
    _stack.push_front(op);
    return (*this);
  }

  /**
   * @brief Create "Load" op and push it onto the internal stack
   *
   * BEFORE| Stack
   * AFTER | Load(obj); Stack
   */
  OpBuilder &load(coco::Object *obj)
  {
    auto op = _module->entity()->op()->create<coco::Load>();
    op->object(obj);
    push(op);
    return (*this);
  }

  /**
   * @brief Create "Add" op and push it onto the internal stack
   *
   * BEFORE| Left; Right; Stack
   * AFTER | Add(Left, Right); Stack
   */
  OpBuilder &add(void) { return binary<coco::Add>(); }

  /**
   * @brief Create "Sub" op and push it onto the internal stack
   *
   * BEFORE| Left; Right; Stack
   * AFTER | Sub(Left, Right); Stack
   */
  OpBuilder &sub(void) { return binary<coco::Sub>(); }

  /**
   * @brief Create "Mul" op and push it onto the internal stack
   *
   * BEFORE| Left; Right; Stack
   * AFTER | Mul(Left, Right); Stack
   */
  OpBuilder &mul(void) { return binary<coco::Mul>(); }

  /**
   * @brief Create "Div" op and push it onto the internal stack
   *
   * BEFORE| Left; Right; Stack
   * AFTER | Div(Left, Right); Stack
   */
  OpBuilder &div(void) { return binary<coco::Div>(); }

  /**
   * @brief Pop op from the internal stack
   *
   * BEFORE| Op; Stack
   * AFTER | Stack
   */
  coco::Op *pop(void)
  {
    assert(_stack.size() > 0);
    auto op = _stack.front();
    _stack.pop_front();
    return op;
  }

private:
  template <typename ConcreteOp> OpBuilder &binary()
  {
    assert(_stack.size() >= 2);
    auto left = pop();
    auto right = pop();

    auto op = _module->entity()->op()->create<ConcreteOp>();
    op->left(left);
    op->right(right);
    push(op);

    return (*this);
  }

private:
  coco::Module *_module;
  std::deque<coco::Op *> _stack;
};

inline OpBuilder op_builder(coco::Module *m) { return OpBuilder{m}; }
inline OpBuilder op_builder(const std::unique_ptr<coco::Module> &m) { return op_builder(m.get()); }

class InstrBuilder
{
public:
  InstrBuilder(coco::Module *module) : _module{module}
  {
    // NOTE _module SHOULD be valid
    assert(_module != nullptr);
  }

public:
  /**
   * @brief Create "Eval" instruction with a given "Object" and "Op"
   *
   * @note "eval(out, op)" will create "%out <- Eval(op)" instruction
   */
  coco::Eval *eval(coco::Object *out, coco::Op *op) const
  {
    auto ins = _module->entity()->instr()->create<coco::Eval>();
    ins->op(op);
    ins->out(out);
    return ins;
  }

private:
  coco::Module *_module;
};

inline InstrBuilder instr_builder(coco::Module *m) { return InstrBuilder{m}; }
inline InstrBuilder instr_builder(const std::unique_ptr<coco::Module> &m)
{
  return instr_builder(m.get());
}

#endif // __IR_BUILDER_H__
