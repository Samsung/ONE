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

#include "coco/IR/Bag.h"

#include "coco/IR/Object.h"
#include "coco/IR/Read.h"
#include "coco/IR/Update.h"

#include <cassert>

namespace coco
{

Bag::Bag(uint32_t size) : _size{size}
{
  // DO NOTHING
}

Bag::~Bag()
{
  // All the references over a bag SHOULD be dropped before its destruction
  assert(deps()->size() == 0);
  assert(reads()->size() == 0);
  assert(updates()->size() == 0);
}

uint32_t Bag::size(void) const { return _size; }

bool Bag::isInput(void) const { return _input != nullptr; }
bool Bag::isOutput(void) const { return _output != nullptr; }

const DepSet *Bag::deps(void) const { return &_deps; }
const ReadSet *Bag::reads(void) const { return &_reads; }
const UpdateSet *Bag::updates(void) const { return &_updates; }

void Bag::replaceWith(Bag *b)
{
  assert(!isInput() && !isOutput());

  replaceAllDepsWith(b);
  // Replace all the occurence inside Read
  while (!(reads()->empty()))
  {
    auto read = *(reads()->begin());
    assert(read->bag() == this);
    read->bag(b);
  }

  // Replace all the occurence insider Update
  while (!(updates()->empty()))
  {
    auto update = *(updates()->begin());
    assert(update->bag() == this);
    update->bag(b);
  }

  assert(deps()->empty());
  assert(reads()->empty());
  assert(updates()->empty());
}

void Bag::replaceAllDepsWith(Bag *b)
{
  // Replace all the occurence inside Dep
  while (!(deps()->empty()))
  {
    auto dep = *(deps()->begin());
    assert(dep->bag() == this);
    dep->bag(b);
  }
}

ObjectSet dependent_objects(const Bag *b)
{
  ObjectSet res;

  for (const auto &dep : *(b->deps()))
  {
    if (auto obj = dep->object())
    {
      res.insert(obj);
    }
  }

  return res;
}

Bag::ReaderSet readers(const Bag *b)
{
  Bag::ReaderSet res;

  for (auto obj : dependent_objects(b))
  {
    for (auto consumer : consumers(obj))
    {
      // NOTE Object::Consumer inherits Bag::Reader
      res.insert(consumer);
    }
  }

  for (auto read : *b->reads())
  {
    auto reader = read->reader();
    assert(reader != nullptr);
    res.insert(reader);
  }

  return res;
}

Bag::UpdaterSet updaters(const Bag *b)
{
  Bag::UpdaterSet res;

  for (auto obj : dependent_objects(b))
  {
    if (auto p = producer(obj))
    {
      res.insert(p);
    }
  }

  for (auto update : *b->updates())
  {
    auto updater = update->updater();
    assert(updater != nullptr);
    res.insert(updater);
  }

  return res;
}

} // namespace coco
