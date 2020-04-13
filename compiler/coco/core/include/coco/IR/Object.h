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

#ifndef __COCO_IR_OBJECT_H__
#define __COCO_IR_OBJECT_H__

#include "coco/IR/Entity.h"
#include "coco/IR/Bag.h"
#include "coco/IR/Dep.h"
#include "coco/IR/Def.forward.h"
#include "coco/IR/UseSet.h"

#include "coco/IR/FeatureObject.forward.h"
#include "coco/IR/KernelObject.forward.h"

#include <set>

namespace coco
{

/**
 * @brief Base interface on all typed NN values
 */
class Object : public Entity
{
public:
  friend class Def;
  friend class Use;

public:
  enum class Kind
  {
    Unknown,
    Feature,
    Kernel,
  };

public:
  struct Producer : public Bag::Updater
  {
    virtual ~Producer() = default;
  };

  struct Consumer : public Bag::Reader
  {
    virtual ~Consumer() = default;
  };

  using ConsumerSet = std::set<Consumer *>;

public:
  Object();

public:
  virtual ~Object() = default;

public:
  virtual Kind kind(void) const { return Kind::Unknown; }

public:
  coco::Bag *bag(void) const { return _dep.bag(); }
  void bag(coco::Bag *bag) { _dep.bag(bag); }

public:
  virtual FeatureObject *asFeature(void) { return nullptr; }
  virtual const FeatureObject *asFeature(void) const { return nullptr; }

  virtual KernelObject *asKernel(void) { return nullptr; }
  virtual const KernelObject *asKernel(void) const { return nullptr; }

public:
  Def *def(void) const;
  const UseSet *uses(void) const;

private:
  /**
   * @brief Update the link to a producer
   *
   * WARN Only Def class is allowed to access this method
   */
  void def(Def *d);

  // NOTE "mutable_" prefix is introduced to avoid resolution issue similarly as in Bag
  // WARN Only Use class is allowed to access this method
  UseSet *mutable_uses(void);

private:
  Dep _dep;
  Def *_def = nullptr;
  UseSet _uses;
};

/**
 * @brief Check whether a given object is of type T
 *
 * The example below shows how to use this "isa<T>" helper:
 *   auto obj = new FeatureObject{};
 *
 *   if (isa<FeatureObject>())
 *   {
 *     std::cout << "FeatureObject" << std::endl;
 *   }
 */
template <typename T> bool isa(const Object *);

/**
 * @brief Cast a generic object as a specific one
 *
 * "cast<T>(o)" accepts only a valid object pointer "o" that "isa<T>(o)" holds
 * - Then, "cast<T>(o)" always returns a valid object pointer.
 */
template <typename T> T *cast(Object *);

/**
 * @brief Cast a generic object as a specific one
 *
 * Unlike "cast<T>", "safe_cast<T>" accepts any object pointer
 * - "safe_cast<T>(nullptr)" returns "nullptr"
 * - "safe_cast<T>(o)" returns "nullptr" if "isa<T>(o)" does not hold
 */
template <typename T> T *safe_cast(Object *);

/// @brief Return the producer of a given object if it exists
Object::Producer *producer(const Object *);

/// @brief Return a set of consumers of a given object.
Object::ConsumerSet consumers(const Object *);

} // namespace coco

#endif // __COCO_IR_OBJECT_H__
