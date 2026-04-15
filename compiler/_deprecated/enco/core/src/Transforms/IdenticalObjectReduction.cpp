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

#include "IdenticalObjectReduction.h"
#include "IRUtils.h"

#include <set>

namespace enco
{

void reduce_identical_object(enco::Code *code)
{
  auto m = code->module();

  std::set<coco::Copy *> detached;

  // Preceding optimizations may generate "free" instructions.
  //  - i.e. an instruction not linked to a block
  //
  // Let's iterate over only a sequence of "bounded" instructions.
  for (auto ins : instr_sequence(m))
  {
    assert(ins != nullptr);
    assert(ins->parent() != nullptr);

    auto copy = ins->asCopy();

    if (copy == nullptr)
    {
      // Skip if instruction is not a copy
      continue;
    }

    // TODO Support non-Feature Objects
    auto ifm = copy->from()->asFeature();
    auto ofm = copy->into()->asFeature();

    assert(ofm->bag() != nullptr);

    if (ifm->layout()->id() != ofm->layout()->id())
    {
      continue;
    }

    if (ifm->layout()->id() != coco::FeatureLayouts::BHWC::uid())
    {
      continue;
    }

    // Skip if this copy produces network output
    if (ofm->bag()->output())
    {
      // TODO Optimize this case
      //
      // Note that the code under optimization is of the following form:
      //
      //   %ifm <- Instr(...)
      //   %ofm <- Copy(%ifm)
      //
      // Let's assume that "Copy" is the only reader of %ifm (to be precise, its bag).
      //
      // Then, it is possible to rewrite the above fragment as follows:
      //
      //   %ofm <- Instr(...)
      //
      continue;
    }

    if (ofm->bag()->reads()->size() > 0)
    {
      // Let us consider the following code:
      //
      // Bag:
      //   %bag_0 = Bag(...)
      //   %bag_1 = Bag(...)
      //   %bag_2 = Bag(...)
      //
      // Object:
      //   %obj_0 = FeatureObject(bag: %bag_0)
      //   %obj_1 = FeatureObject(bag: %bag_1)
      //
      // Instr:
      //   copy an object from %obj_0 into %obj_1
      //   shuffle values from %bag_1 into %bag_2
      //   eval Conv2D with %obj_1
      //
      // Identical Object Reduction (IOR) tries to eliminate the first copy via
      // substitution (substitute all the occurrence of %obj_1 as use with %obj_0).
      //
      // Here is the code transformed by IOR:
      //
      // Bag:
      //   %bag_0 = Bag(...)
      //   %bag_1 = Bag(...)
      //   %bag_2 = Bag(...)
      //
      // Object:
      //   %obj_0 = FeatureObject(bag: %bag_0)
      //   %obj_1 = FeatureObject(bag: %bag_1)
      //
      // Instr:
      //   shuffle values from %bag_1 into %bag_2
      //   eval Conv2D with %obj_0
      //
      // Note that there is no updater of %bag_1 after IOR, and thus the behavior
      // of the first shuffle instruction has changed.
      //
      // This examples shows that it is impossible to simply substitute %obj_1
      // with %obj_0 in the presence of readers over its backing bag.
      continue;
    }

    subst(copy->into(), copy->from());

    copy->detach();
    detached.insert(copy);
  }

  for (auto copy : detached)
  {
    m->entity()->instr()->destroy(copy);
  }
}

} // namespace enco
