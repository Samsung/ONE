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

#ifndef __ENCO_TRANSFORM_DUPLICATED_OBJECT_REDUCTION_H__
#define __ENCO_TRANSFORM_DUPLICATED_OBJECT_REDUCTION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Reduce duplicated feature objects as its dominating feature object
 *
 * >>> BEFORE <<<
 * %obj_0 = Feature(layout: ???) at ...
 * %obj_1 = Feature(layout: BHWC) at ...
 * %obj_2 = Feature(layout: BHWC) at ...
 *
 * copy(from: %obj_0, into: %obj_1)
 * copy(from: %obj_0, into: %obj_2)
 *
 * ...
 * Use(%obj_1)
 * Use(%obj_2)
 * ...
 *
 * >>> AFTER <<<
 * %obj_0 = Feature(layout: ???) at ...
 * %obj_1 = Feature(layout: BHWC) at ...
 * %obj_2 = Feature(layout: BHWC) at ...
 *
 * copy(from: %obj_0, into: %obj_1)
 * copy(from: %obj_0, into: %obj_2)
 *
 * ...
 * Use(%obj_1)
 * Use(%obj_1) <-- CHANGED
 * ...
 *
 * NOTE Given a set of feature objects, a feature object referred to as a dominating
 *      feature object if its producer proceeds the producer of every feature object
 *      in the given set
 */
void reduce_duplicated_object(enco::Code *code);

struct DuplicatedObjectReductionPass final : public Pass
{
  PASS_CTOR(DuplicatedObjectReductionPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { reduce_duplicated_object(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_DUPLICATED_OBJECT_REDUCTION_H__
