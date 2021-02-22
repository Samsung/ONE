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

#include "coco/IR/Module.h"

#include <memory>

using std::make_unique;

namespace
{

struct EntityManagerImpl final : public coco::EntityManager
{
public:
  std::unique_ptr<coco::BagManager> _bag;

public:
  coco::BagManager *bag(void) override { return _bag.get(); }
  const coco::BagManager *bag(void) const override { return _bag.get(); }

public:
  std::unique_ptr<coco::ObjectManager> _object;

public:
  coco::ObjectManager *object(void) override { return _object.get(); }
  const coco::ObjectManager *object(void) const override { return _object.get(); }

public:
  std::unique_ptr<coco::OpManager> _op;

public:
  coco::OpManager *op(void) override { return _op.get(); }
  const coco::OpManager *op(void) const override { return _op.get(); }

public:
  coco::InstrManager *instr(void) override { return _instr.get(); }
  const coco::InstrManager *instr(void) const override { return _instr.get(); }

public:
  coco::BlockManager *block(void) override { return _block.get(); }
  const coco::BlockManager *block(void) const override { return _block.get(); }

public:
  std::unique_ptr<coco::InputManager> _input;

public:
  coco::InputManager *input(void) override { return _input.get(); }
  const coco::InputManager *input(void) const override { return _input.get(); }

public:
  std::unique_ptr<coco::OutputManager> _output;

public:
  coco::OutputManager *output(void) override { return _output.get(); }
  const coco::OutputManager *output(void) const override { return _output.get(); }

public:
  // WARN Do NOT change the order of these fields: _block -> _instr
  //
  // Note that each instruction may have a reference to a block, and
  // the destructor of Instr accesses this 'block' reference.
  //
  // Thus, Instr entities SHOULD BE destructed before Block entities are destructed.
  std::unique_ptr<coco::BlockManager> _block;
  std::unique_ptr<coco::InstrManager> _instr;
};

} // namespace

namespace
{

class ModuleImpl final : public coco::Module
{
public:
  coco::EntityManager *entity(void) override { return _entity.get(); }
  const coco::EntityManager *entity(void) const override { return _entity.get(); }

public:
  std::unique_ptr<coco::BlockList> _block;

public:
  coco::BlockList *block(void) override { return _block.get(); }
  const coco::BlockList *block(void) const override { return _block.get(); }

public:
  std::unique_ptr<coco::InputList> _input;

public:
  coco::InputList *input(void) override { return _input.get(); }
  const coco::InputList *input(void) const override { return _input.get(); }

public:
  std::unique_ptr<coco::OutputList> _output;

public:
  coco::OutputList *output(void) override { return _output.get(); }
  const coco::OutputList *output(void) const override { return _output.get(); }

public:
  // WARN _entity SHOULD BE declared after _block in order to allow each Block(s) to detach itself.
  //
  // If not, Block is destructed after its corresponding BlockList is destructed, which results
  // in invalid memory access during the update on BlockList (inside Block's destructor).
  std::unique_ptr<coco::EntityManager> _entity;
};

} // namespace

namespace coco
{

std::unique_ptr<Module> Module::create(void)
{
  auto m = make_unique<::ModuleImpl>();

  auto mgr = make_unique<::EntityManagerImpl>();
  {
    mgr->_bag = make_unique<coco::BagManager>(m.get());
    mgr->_object = make_unique<coco::ObjectManager>(m.get());
    mgr->_op = make_unique<coco::OpManager>(m.get());
    mgr->_instr = make_unique<coco::InstrManager>(m.get());
    mgr->_block = make_unique<coco::BlockManager>(m.get());
    mgr->_input = make_unique<coco::InputManager>(m.get());
    mgr->_output = make_unique<coco::OutputManager>(m.get());
  }
  m->_entity = std::move(mgr);

  m->_block = make_unique<coco::BlockList>(m.get());
  m->_input = make_unique<coco::InputList>();
  m->_output = make_unique<coco::OutputList>();

  return std::move(m);
}

} // namespace coco
