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

#ifndef __COCO_IR_BAG_H__
#define __COCO_IR_BAG_H__

#include "coco/IR/Entity.h"
#include "coco/IR/ObjectSet.h"
#include "coco/IR/DepSet.h"
#include "coco/IR/ReadSet.h"
#include "coco/IR/UpdateSet.h"
#include "coco/IR/Input.forward.h"
#include "coco/IR/Output.forward.h"
#include "coco/IR/Locatable.h"

#include <set>

#include <memory>

namespace coco
{

/**
 * @brief A collection of (abstracted) elements of the same type
 *
 * When there are N elements in a bag, we refer to N as the size of this bag, and every
 * element in a bag has a unique numeric ID whose range is [0, N).
 *
 * NOTE 'Bag' is not a container (such as std::vector). 'Bag' just assures that there are
 *      N elements. It does not state about its value.
 *
 * NOTE coco IR treats Bag as virtual memory allocation
 */
class Bag final : public Entity
{
public:
  struct Updater : public Locatable
  {
    virtual ~Updater() = default;
  };

  using UpdaterSet = std::set<Updater *>;

  struct Reader : public Locatable
  {
    virtual ~Reader() = default;
  };

  using ReaderSet = std::set<Reader *>;

public:
  friend class Dep;
  friend class Read;
  friend class Update;
  friend class Input;
  friend class Output;

public:
  explicit Bag(uint32_t size);

public:
  ~Bag();

public:
  uint32_t size(void) const;

public:
  bool isInput(void) const;
  bool isOutput(void) const;

public:
  /// @brief Return the set of Dep links that point to this bag
  const DepSet *deps(void) const;
  /// @brief Return the set of Read links that point to this bag
  const ReadSet *reads(void) const;
  /// @brief Return the set of Update links that point to this bag
  const UpdateSet *updates(void) const;

public:
  /// @brief Return a valid pointer if this bag is marked as an input of the model
  Input *input(void) const { return _input; }
  /// @brief Return a valid pointer if this bag is marked as an output of the model
  Output *output(void) const { return _output; }

public:
  /**
   * @brief Replace all the occurence of a bag (except those in Input/Output) with another bag
   *
   * NOTE reaplceWith(b) works correctly only when b is neither Input nor Output
   */
  void replaceWith(Bag *b);

  /**
   * @brief Replace all the occurence of a bag in Object with another bag
   *
   * NOTE Unlike replaceWith(b), replaceAllDepsWith(b) has no restriction
   */
  void replaceAllDepsWith(Bag *);

private:
  // "mutable_" prefix is deliberately introduced below to avoid resolution issue.
  //
  // Let's assume that two "deps" are overloaded in Bag as follows:
  // class Bag
  // {
  // private:
  //   DepSet *deps(void); <-- 1
  // public:
  //   const DepSet *deps(void) const; <-- 2
  // };
  //
  // C++ compiler tries to invoke method 1 unless a bag itself is const. Thus, any "deps" calls
  // over non-const bags except those calls from friend classes will introduce build error.

  // WARN Only Dep is allowed to access this method
  DepSet *mutable_deps(void) { return &_deps; }
  // WARN Only Read is allowed to access this method
  ReadSet *mutable_reads(void) { return &_reads; }
  // WARN Only Update is allowed to access this method
  UpdateSet *mutable_updates(void) { return &_updates; }

private:
  // WARN Only Input is allowed to access this method
  void input(Input *i) { _input = i; }
  // WARN Only Output is allowed to access this method
  void output(Output *o) { _output = o; }

private:
  uint32_t _size;

  /** @brief Links to dependent Object(s) */
  DepSet _deps;
  /** @brief Direct reads (not through Object) */
  ReadSet _reads;
  /** @brief Direct updates (not through Object) */
  UpdateSet _updates;

  Input *_input = nullptr;
  Output *_output = nullptr;
};

/// @brief Return a set of objects that depends on a given bag
ObjectSet dependent_objects(const Bag *);
/// @brief Return a set of readers that reads a given bag
Bag::ReaderSet readers(const Bag *);
/// @brief Return a set of updaters that updates a given bag
Bag::UpdaterSet updaters(const Bag *);

} // namespace coco

#endif // __COCO_IR_BAG_H__
