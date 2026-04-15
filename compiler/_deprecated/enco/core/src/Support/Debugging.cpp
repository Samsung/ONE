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

#include "Debugging.h"

#include <pp/LinearDocument.h>
#include <pp/MultiLineTextUtils.h>

#include <stack>

#include <iostream>

#define DEBUGGING_API_P(NAME, TYPE, VAR)                         \
  static void _##NAME(const TYPE *);                             \
  void NAME(long p) { NAME(reinterpret_cast<const TYPE *>(p)); } \
  void NAME(const TYPE *p)                                       \
  {                                                              \
    if (p == nullptr)                                            \
    {                                                            \
      std::cout << "(nullptr)" << std::endl;                     \
    }                                                            \
    else                                                         \
    {                                                            \
      _##NAME(p);                                                \
    }                                                            \
  }                                                              \
  void _##NAME(const TYPE *VAR)

namespace
{

class SectionBuilder
{
public:
  SectionBuilder(const std::string &tag) : _tag{tag}
  {
    // DO NOTHING
  }

public:
  template <typename Callback> pp::LinearDocument build(Callback cb) const
  {
    pp::LinearDocument res;

    res.append(_tag, " {");
    res.indent();

    cb(res);

    res.unindent();
    res.append("}");

    return res;
  }

private:
  std::string _tag;
};

template <typename Callback>
pp::LinearDocument operator<<(const SectionBuilder &builder, Callback cb)
{
  return builder.build(std::forward<Callback>(cb));
}

SectionBuilder section(const std::string &tag) { return SectionBuilder{tag}; }
} // namespace

/**
 * SECTION: Bag
 */
namespace
{

pp::LinearDocument describe(const coco::Bag *bag)
{
  pp::LinearDocument doc;

  doc.append("addr: ", bag);
  doc.append("size: ", bag->size());
  // TODO Print Read
  // TODO Print Update
  // TODO Print Dep
  return doc;
}

} // namespace

DEBUGGING_API_P(enco_dump_all_bags, coco::Module, m)
{
  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);
    assert(bag != nullptr);

    auto set = [bag](pp::LinearDocument &doc) { doc.append(describe(bag)); };
    auto desc = section("bag").build(set);

    std::cout << desc << std::endl;
  }
}

/**
 * SECTION: Object
 */
namespace
{
std::string op_kind(const coco::Op *op);

/**
 * @brief Return the def(producer) type of object
 */
std::string def_kind(const coco::Def *def)
{
  if (def)
  {
    if (auto instr = dynamic_cast<coco::Instr *>(def->producer()))
    {
      std::stringstream ss;

      if (auto eval = instr->asEval())
      {
        ss << op_kind(eval->op()) << "(" << instr << ")";
        return ss.str();
      }
      else if (instr->asCopy())
      {
        ss << "Copy(" << instr << ")";
        return ss.str();
      }
      else if (instr->asShuffle())
      {
        ss << "Shuffle(" << instr << ")";
        return ss.str();
      }
    }
    else
    {
      return "(unknown)";
    }
  }

  return "(none)";
}

pp::LinearDocument describe(const coco::Object *obj)
{
  pp::LinearDocument doc;

  doc.append("addr: ", obj);
  doc.append("bag: ", obj->bag());
  doc.append("producer: ", def_kind(obj->def()));
  // TODO Show Uses
  // TODO Show FeatureObject/KernelObect info

  return doc;
}

} // namespace

DEBUGGING_API_P(enco_dump_all_objects, coco::Module, m)
{
  for (uint32_t n = 0; n < m->entity()->object()->size(); ++n)
  {
    auto obj = m->entity()->object()->at(n);
    assert(obj != nullptr);

    auto set = [obj](pp::LinearDocument &doc) { doc.append(describe(obj)); };
    auto desc = section("object").build(set);

    std::cout << desc << std::endl;
  }
}

/**
 * SECTION: Op
 */
namespace
{

struct OpTree
{
public:
  OpTree(const coco::Op *op) : _op{op}
  {
    // DO NOTHING
  }

public:
  const coco::Op *root(void) const { return _op; }

private:
  const coco::Op *_op;
};

std::string op_kind(const coco::Op *op)
{
  struct OpKind : public coco::Op::Visitor<std::string>
  {
    std::string visit(const coco::Load *) override { return "Load"; }
    std::string visit(const coco::Conv2D *) override { return "Conv2D"; }
    std::string visit(const coco::MaxPool2D *) override { return "MaxPool2D"; }
    std::string visit(const coco::AvgPool2D *) override { return "AvgPool2D"; }
    std::string visit(const coco::PadF *) override { return "PadF"; }
    std::string visit(const coco::ReLU *) override { return "ReLU"; }
    std::string visit(const coco::Add *) override { return "Add"; }
    std::string visit(const coco::Mul *) override { return "Mul"; }
    std::string visit(const coco::ConcatF *) override { return "ConcatF"; }
    std::string visit(const coco::Sub *) override { return "Sub"; }
    std::string visit(const coco::Sqrt *) override { return "Sqrt"; }
    std::string visit(const coco::Div *) override { return "Div"; }
  };

  OpKind v;

  return op->accept(v);
}

pp::LinearDocument describe(const coco::Padding2D *pad)
{
  pp::LinearDocument doc;

  doc.append("top: ", pad->top());
  doc.append("bottom: ", pad->bottom());
  doc.append("left: ", pad->left());
  doc.append("right: ", pad->right());

  return doc;
}

pp::LinearDocument describe(const coco::Stride2D *stride)
{
  pp::LinearDocument doc;

  doc.append("vertical: ", stride->vertical());
  doc.append("horizontal ", stride->horizontal());

  return doc;
}

pp::LinearDocument describe(const coco::Conv2D *conv)
{
  pp::LinearDocument doc;

  doc.append("arg: ", conv->arg());
  doc.append("ker: ", conv->ker());
  doc.append("group: ", conv->group());

  if (auto pad = conv->pad())
  {
    auto set = [pad](pp::LinearDocument &doc) { doc.append(describe(pad)); };
    auto desc = section("pad").build(set);
    doc.append(desc);
  }

  if (auto stride = conv->stride())
  {
    auto set = [stride](pp::LinearDocument &doc) { doc.append(describe(stride)); };
    auto desc = section("stride").build(set);
    doc.append(desc);
  }

  return doc;
}

pp::LinearDocument describe(const coco::Op *op)
{
  pp::LinearDocument doc;

  doc.append("addr: ", op);
  doc.append("kind: ", op_kind(op));
  doc.append("parent(instr): ", op->parent());
  doc.append("up(op): ", op->up());

  if (auto conv = op->asConv2D())
  {
    auto set = [conv](pp::LinearDocument &doc) { doc.append(describe(conv)); };
    auto desc = section("conv2d").build(set);
    doc.append(desc);
  }
  else if (auto load = op->asLoad())
  {
    auto set = [load](pp::LinearDocument &doc) { doc.append(describe(load->object())); };
    auto desc = section("load").build(set);
    doc.append(desc);
  }

  return doc;
}

pp::LinearDocument describe(const OpTree &t, bool verbose = false)
{
  pp::LinearDocument doc;

  struct Frame
  {
  public:
    Frame(const coco::Op *op) : _op{op}, _indicator{0}
    {
      // op SHOULD BE valid
      assert(_op != nullptr);
    }

  public:
    /**
     * @brief Return a pointer to coco::Op of interest
     */
    const coco::Op *op(void) const { return _op; }

    /**
     * @brief Return the indicator
     *
     * Let's assume that the arity of a coco::Op of interest is N
     *  INDICATOR 0     -> Print the op itself
     *  INDICATOR 1     -> Print the first argument
     *  ...
     *  INDICATOR N     -> Print the N-th argument
     *  INDICATOR N + 1 -> Done
     */
    uint32_t indicator(void) const { return _indicator; }

  public:
    void advance(void) { _indicator += 1; }

  private:
    const coco::Op *_op;
    uint32_t _indicator;
  };

  std::stack<Frame> stack;

  stack.emplace(t.root());

  while (stack.size() > 0)
  {
    auto op = stack.top().op();
    uint32_t indicator = stack.top().indicator();

    if (indicator == 0)
    {
      doc.append(op_kind(op), " (", op, ")");

      doc.indent();
      stack.top().advance();

      // TODO Need to update it to better design for verbose flag
      if (verbose)
      {
        auto set = [op](pp::LinearDocument &doc) { doc.append(describe(op)); };
        auto desc = section("op").build(set);
        doc.append(desc);
      }
    }
    else if (indicator < op->arity() + 1)
    {
      stack.top().advance();
      stack.emplace(op->arg(indicator - 1));
    }
    else
    {
      assert(indicator == op->arity() + 1);
      doc.unindent();
      stack.pop();
    }
  }

  return doc;
}

} // namespace

DEBUGGING_API_P(enco_dump_op, coco::Op, op)
{
  {
    std::cout << describe(op) << std::endl;
  }
}

DEBUGGING_API_P(enco_dump_op_tree, coco::Op, op)
{
  {
    std::cout << describe(OpTree(op)) << std::endl;
  }
}

DEBUGGING_API_P(enco_dump_all_ops, coco::Module, m)
{
  SectionBuilder section_builder{"op"};

  for (uint32_t n = 0; n < m->entity()->op()->size(); ++n)
  {
    auto op = m->entity()->op()->at(n);
    assert(op != nullptr);

    auto desc = section("op").build([op](pp::LinearDocument &doc) { doc.append(describe(op)); });

    std::cout << desc << std::endl;
  }
}

/**
 * SECTION: Instr
 */
namespace
{

std::string kind(const coco::Instr *ins)
{
  struct InstrKind : public coco::Instr::Visitor<std::string>
  {
    std::string visit(const coco::Eval *) override { return "Eval"; }
    std::string visit(const coco::Copy *) override { return "Copy"; }
    std::string visit(const coco::Shuffle *) override { return "Shuffle"; }
  };

  InstrKind v;

  return ins->accept(v);
}

pp::LinearDocument describe(const coco::Instr *ins, bool verbose = false)
{
  pp::LinearDocument doc;

  doc.append("addr: ", ins);
  doc.append("kind: ", kind(ins));
  doc.append("parent: ", ins->parent());

  // TODO Need to update it to better design for verbose flag
  if (verbose)
  {
    if (auto eval = ins->asEval())
    {
      auto optset = [eval, verbose](pp::LinearDocument &doc) {
        doc.append(describe(OpTree(eval->op()), verbose));
      };
      auto optdesc = section("op").build(optset);
      doc.append(optdesc);

      auto outset = [eval](pp::LinearDocument &doc) { doc.append(describe(eval->out())); };
      auto outdesc = section("out").build(outset);
      doc.append(outdesc);
    }
    else if (auto copy = ins->asCopy())
    {
      auto from = [copy](pp::LinearDocument &doc) { doc.append(describe(copy->from())); };
      auto into = [copy](pp::LinearDocument &doc) { doc.append(describe(copy->into())); };

      auto fdesc = section("from").build(from);
      doc.append(fdesc);

      auto idesc = section("into").build(into);
      doc.append(idesc);
    }
  }

  return doc;
}

} // namespace

DEBUGGING_API_P(enco_dump_all_instrs, coco::Module, m)
{
  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    auto ins = m->entity()->instr()->at(n);
    assert(ins != nullptr);

    auto setter = [ins](pp::LinearDocument &doc) { doc.append(describe(ins)); };
    auto desc = section("instr").build(setter);

    std::cout << desc << std::endl;
  }
}

DEBUGGING_API_P(enco_dump_all_instrs_v, coco::Module, m)
{
  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    auto ins = m->entity()->instr()->at(n);
    assert(ins != nullptr);

    auto setter = [ins](pp::LinearDocument &doc) { doc.append(describe(ins, true)); };
    auto desc = section("instr").build(setter);

    std::cout << desc << std::endl;
  }
}

DEBUGGING_API_P(enco_dump_instr, coco::Instr, ins)
{
  auto setter = [ins](pp::LinearDocument &doc) { doc.append(describe(ins, true)); };
  auto desc = section("instr").build(setter);

  std::cout << desc << std::endl;
}

/**
 * SECTION: Block
 */
namespace
{

pp::LinearDocument describe(const coco::Block *blk)
{
  pp::LinearDocument doc;

  for (auto ins = blk->instr()->head(); ins; ins = ins->next())
  {
    auto setter = [ins](pp::LinearDocument &doc) { doc.append(describe(ins)); };
    auto desc = section("instr").build(setter);
    doc.append(desc);
  }

  return doc;
}

} // namespace

DEBUGGING_API_P(enco_dump_block, coco::Block, blk) { std::cout << describe(blk) << std::endl; }
