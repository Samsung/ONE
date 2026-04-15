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

#ifndef __ENCO_TRANSFORM_IDENTICAL_OBJECT_REDUCTION_H__
#define __ENCO_TRANSFORM_IDENTICAL_OBJECT_REDUCTION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Reduce identically copied objects as its original object
 *
 * >>> BEFORE <<<
 * %bag_0 = Bag(size: N)
 * %bag_1 = Bag(size: N)
 *
 * %obj_0 = Feature(layout: BHWC) at %bag_0
 * %obj_1 = Feature(layout: BHWC) at %bag_1
 *
 * copy(from: %obj_0, into: %obj_1)
 * ...
 * Use(%obj_0)
 * Use(%obj_1)
 * ...
 *
 * >>> AFTER <<<
 * %bag_0 = Bag(size: N)
 * %bag_1 = Bag(size: N)
 *
 * %obj_0 = Feature(layout: BHWC) at %bag_0
 * %obj_1 = Feature(layout: BHWC) at %bag_1
 *
 * copy(from: %obj_0, into: %obj_1)
 * ...
 * Use(%obj_0)
 * Use(%obj_0) <- %obj_1 is replaced
 * ...
 */
void reduce_identical_object(enco::Code *code);

struct IdenticalObjectReductionPass final : public Pass
{
  PASS_CTOR(IdenticalObjectReductionPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { reduce_identical_object(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_IDENTICAL_OBJECT_REDUCTION_H__
