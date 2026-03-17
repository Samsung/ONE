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
 * @file Dump.cpp
 * @brief Print coco IR produced from enco frontend
 *
 * @note  Some object inherits multiple parents.
 * For example, coco:Conv2D inherits coco::Consumer and more. Assume that op is an instance
 * of coco::Conv2D. In this case, the printing results of the following may be different:
 *     1) cout << op; // printing address of type coco::Conv2D
 *     2) cout << reinterpret_cast<const coco::Object::Consumer *>(op);
 *     3) cout << object->consumer(); // assume that this object->consumer() returns op
 *     4) cout << dynamic_cast<const coco::Object::Consumer *>(op);
 *  1) and 2) prints same address. 3) and 4) prints same address but different from 1) and 2).
 * For details, refer to
 * https://stackoverflow.com/questions/22256620/why-pointers-to-the-same-object-have-different-values
 * For dumping, we will use 3), 4)
 */
#include "Dump.h"

#include <functional>
#include <iostream>

std::string tab(int n) { return std::string(n * 2, ' '); }

struct OpPrinter final : public coco::Op::Visitor<void>
{
public:
  OpPrinter(std::ostream &os, int indent) : _os(os), _indent(indent) {}

public:
  void visit(const coco::Load *op) override
  {
    _os << tab(_indent) << "Load(" << dynamic_cast<const coco::Op *>(op)
        << ", obj: " << op->object() << ")" << std::endl;
  }

  void visit(const coco::PadF *op) override
  {
    op->arg()->accept(this);
    _os << tab(_indent) << "PadF" << std::endl;
  }

  void visit(const coco::Conv2D *op) override
  {
    op->arg()->accept(this);
    const coco::Padding2D *pad = op->pad();
    const coco::Stride2D *stride = op->stride();

    _os << tab(_indent) << "Conv2D(" << dynamic_cast<const coco::Op *>(op)
        << ", ker obj: " << op->ker() << ", padding [T/B/L/R=" << pad->top() << "," << pad->bottom()
        << "," << pad->left() << "," << pad->right() << "]"
        << ", stride [V/H = " << stride->vertical() << "," << stride->horizontal() << "]"
        << ")" << std::endl;
  }

  void visit(const coco::MaxPool2D *op) override
  {
    op->arg()->accept(this);
    _os << tab(_indent) << "MaxPool2D" << std::endl;
  }

  void visit(const coco::AvgPool2D *op) override
  {
    op->arg()->accept(this);
    _os << tab(_indent) << "AvgPool2D" << std::endl;
  }

  void visit(const coco::Add *op) override
  {
    op->left()->accept(this);
    op->right()->accept(this);
    _os << tab(_indent) << "Add" << std::endl;
  }

  void visit(const coco::Mul *op) override
  {
    op->left()->accept(this);
    op->right()->accept(this);
    _os << tab(_indent) << "Mul" << std::endl;
  }

  void visit(const coco::ReLU *op) override
  {
    op->arg()->accept(this);
    _os << tab(_indent) << "ReLU" << std::endl;
  }

  void visit(const coco::ReLU6 *op) override
  {
    op->arg()->accept(this);
    _os << tab(_indent) << "ReLU6" << std::endl;
  }

  void visit(const coco::Sub *op) override
  {
    op->left()->accept(this);
    op->right()->accept(this);
    _os << tab(_indent) << "Sub" << std::endl;
  }

  void visit(const coco::ConcatF *op) override
  {
    op->left()->accept(this);
    op->right()->accept(this);
    _os << tab(_indent) << "ConcatF" << std::endl;
  }

  void visit(const coco::Div *op) override
  {
    op->left()->accept(this);
    op->right()->accept(this);
    _os << tab(_indent) << "Div" << std::endl;
  }

private:
  std::ostream &_os;

private:
  int _indent;
};

struct InstrPrinter final : public coco::Instr::Visitor<void>
{
public:
  InstrPrinter() = delete;

  InstrPrinter(int indent) : _indent(indent) {}

  void visit(const coco::Eval *ins) override
  {
    std::cout << tab(_indent) << "Eval (" << dynamic_cast<const coco::Instr *>(ins) << ")"
              << std::endl;
    std::cout << tab(_indent + 1) << "out: " << ins->out() << std::endl;
    std::cout << tab(_indent + 1) << "<op>: " << std::endl;
    {
      OpPrinter prn(std::cout, _indent + 2);
      ins->op()->accept(prn);
    }
  }

  void visit(const coco::Copy *ins) override
  {
    // copy is Producer and also Customer. We will use address for Producer
    std::cout << tab(_indent) << "Copy (" << dynamic_cast<const coco::Instr *>(ins) << ")"
              << std::endl;
    std::cout << tab(_indent) << "  from: " << ins->from() << std::endl;
    std::cout << tab(_indent) << "  into: " << ins->into() << std::endl;
  }

  void visit(const coco::Shuffle *ins) override
  {
    std::cout << tab(_indent) << "Shuffle (" << dynamic_cast<const coco::Instr *>(ins) << ")"
              << std::endl;
    std::cout << tab(_indent) << "  from: " << ins->from() << std::endl;
    std::cout << tab(_indent) << "  into: " << ins->into() << std::endl;
  }

private:
  int _indent;
};

void dump(const coco::Op *op, int indent)
{
  OpPrinter prn(std::cout, indent);
  op->accept(prn);
}

void dump(const coco::Instr *ins, int indent)
{
  std::cout << tab(indent) << "<Inst>:" << std::endl;

  static InstrPrinter prn(indent + 1);

  ins->accept(prn);
}

void dump(const coco::Block *B, int indent)
{
  std::cout << tab(indent) << "<Block> (index: " << B->index().value() << ")" << std::endl;
  for (auto I = B->instr()->head(); I != nullptr; I = I->next())
  {
    dump(I, indent + 1);
  }
}

void dump(const coco::BlockList *L, int indent)
{
  for (auto B = L->head(); B != nullptr; B = B->next())
  {
    dump(B, indent);
  }
}

template <typename SetT, typename EntityF>
void dump(std::string header, SetT set, EntityF print_addr_f)
{
  std::cout << header << ": [";
  if (set->size() == 0)
    std::cout << "x";
  else
  {
    int idx = 0;
    for (auto entity : *set)
    {
      if (idx++ != 0)
        std::cout << ", ";
      print_addr_f(entity);
    }
  }
  std::cout << "]";
}

void dump(const coco::BagManager *l, int indent)
{
  std::cout << tab(indent) << "<Bag>:" << std::endl;

  for (auto n = 0; n < l->size(); ++n)
  {
    auto bag = l->at(n);

    std::cout << tab(indent + 1) << bag << ", ";

    // print objects in bag->deps()
    auto print_dep_object = [](coco::Dep *dep) { std::cout << dep->object(); };
    dump("obj", bag->deps(), print_dep_object);
    std::cout << ", ";

    std::cout << "size: " << bag->size() << ", ";

    if (bag->isInput())
      std::cout << "input, ";
    if (bag->isOutput())
      std::cout << "output, ";
    if ((!bag->isInput()) || (!bag->isOutput()))
      std::cout << "const, ";

    // print readers in bag->reads()
    auto print_read_reader = [](coco::Read *read) {
      if (coco::Op *op = dynamic_cast<coco::Op *>(read->reader()))
        std::cout << "op: " << op;
      else if (coco::Instr *instr = dynamic_cast<coco::Instr *>(read->reader()))
        std::cout << "instr: " << instr;
      else
        std::cout << "x";
    };
    dump("reader", bag->reads(), print_read_reader);
    std::cout << ", ";

    // print updaters in bag->updates()
    auto print_update_updater = [](coco::Update *update) {
      if (coco::Op *op = dynamic_cast<coco::Op *>(update->updater()))
        std::cout << "op: " << op;
      else if (coco::Instr *instr = dynamic_cast<coco::Instr *>(update->updater()))
        std::cout << "instr: " << instr;
      else
        std::cout << "x";
    };
    dump("updater", bag->updates(), print_update_updater);
    std::cout << ", ";

    std::cout << std::endl;
  }
}

void dump(coco::FeatureObject *feature_ob)
{
  auto shape = feature_ob->shape();
  std::cout << "kind: Feature, Shape [H/W/D=" << shape.height() << "," << shape.width() << ","
            << shape.depth() << "]";
}

void dump(coco::KernelObject *kernel_ob)
{
  auto shape = kernel_ob->shape();
  std::cout << "kind: Kernel, Shape [N/H/W/D=" << shape.count() << "," << shape.height() << ","
            << shape.width() << "," << shape.depth() << "]";
}

void dump(const coco::ObjectManager *l, int indent)
{
  std::cout << tab(indent) << "<Object>:" << std::endl;
  for (auto n = 0; n < l->size(); ++n)
  {
    auto obj = l->at(n);
    std::cout << tab(indent + 1) << obj << ", bag: " << obj->bag() << ", ";

    using ObDumpers = std::function<void(coco::Object * ob)>;

    std::map<coco::Object::Kind, ObDumpers> ob_dumpers;

    ob_dumpers[coco::Object::Kind::Feature] = [](coco::Object *ob) { dump(ob->asFeature()); };
    ob_dumpers[coco::Object::Kind::Kernel] = [](coco::Object *ob) { dump(ob->asKernel()); };
    ob_dumpers[coco::Object::Kind::Unknown] = [](coco::Object *ob) {
      std::cout << "kind: Unknown";
    };

    ob_dumpers[obj->kind()](obj);

    std::cout << ", producer: ";
    auto def = obj->def();
    if (def)
    {
      if (coco::Op *op = dynamic_cast<coco::Op *>(def->producer()))
        std::cout << "op: " << op;
      else if (coco::Instr *instr = dynamic_cast<coco::Instr *>(def->producer()))
        std::cout << "instr: " << instr;
      else
        std::cout << "x";
    }
    else
      std::cout << "x";
    std::cout << ", ";

    // print consumers in obj->uses()
    auto print_consumer = [](coco::Use *use) {
      if (coco::Op *op = dynamic_cast<coco::Op *>(use->consumer()))
        std::cout << "op: " << op;
      else if (coco::Instr *instr = dynamic_cast<coco::Instr *>(use->consumer()))
        std::cout << "inst: " << instr;
      else
        std::cout << "x";
    };
    dump("comsumer", obj->uses(), print_consumer);
    std::cout << std::endl;
  }
}

template <typename T> void head(int indent);

template <> void head<coco::Input>(int indent) { std::cout << tab(indent) << "<Input>: "; }

template <> void head<coco::Output>(int indent) { std::cout << tab(indent) << "<Output>: "; }

template <typename PtrItemT> void dump(const coco::PtrList<PtrItemT> *list, int indent)
{
  head<PtrItemT>(indent);
  for (int n = 0; n < list->size(); n++)
  {
    const PtrItemT *item = list->at(n);
    if (n != 0)
      std::cout << ", ";
    std::cout << "bag " << item->bag() << ", name=" << item->name();
  }
  std::cout << std::endl;
}

void dump(const coco::Module *module)
{
  std::cout << "<Module>" << std::endl;

  dump(module->block(), 1);
  dump(module->input(), 1);
  dump(module->output(), 1);
  dump(module->entity()->bag(), 1);
  dump(module->entity()->object(), 1);
}
