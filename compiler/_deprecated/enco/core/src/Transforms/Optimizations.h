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

#ifndef __ENCO_OPTIMIZATIONS_H__
#define __ENCO_OPTIMIZATIONS_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Add a bypass Shuffle if two continued Shuffles map same from-into
 *
 * %bag_1 = Bag(size: N)
 * %bag_2 = Bag(size: N)
 * %bag_3 = Bag(size: N)
 *
 * >>> BEFORE <<<
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0])
 * Shuffle(from: %bag_2, into: %bag_3, [0 -> 0])
 *
 * Let's refer to the former shuffle as Shuffle 1 and the latter one as Shuffle 2.
 * We can replace Shuffle 2 with new Shuffle 3 as follows when Shuffle 1 and
 * Shuffle 2 map to the same position.
 *
 * >>> AFTER <<<
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0]) <- Shuffle 1
 * Shuffle(from: %bag_1, into: %bag_3, [0 -> 0]) <- Shuffle 3
 *
 * Note that Shuffle 1 can be eliminated when %bag_2 is not used
 */
void generate_bypass_shuffle(enco::Code *code);

struct BypassGenerationPass final : public Pass
{
  PASS_CTOR(BypassGenerationPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { generate_bypass_shuffle(code(sess)); }
};

/**
 * @brief Update the base bag of each object if possible
 *
 * --- Case 1 ---
 * Let us consider the following code:
 *
 * %bag_1 = Bag(size: 4)
 * %bag_2 = Bag(size: 1)
 *
 * %obj_1 = ... at %bag_1
 * %obj_2 = ... at %bag_2
 *
 * ...
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0]) <- shuffle
 * ...
 *
 * Note that the content of %bag_2 after shuffle is identical to a part of %bag_1, so
 * the following code is identical to the above code
 *
 * %bag_1 = Bag(size: 4)
 * %bag_2 = Bag(size: 1)
 *
 * %obj_1 = ... at %bag_1
 * %obj_2 = ... at %bag_1
 *
 * ...
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0])
 * ...
 *
 * --- Case 2 ---
 * Let us consider the following code:
 *
 * %bag_1 = Bag(size: 4)
 * %bag_2 = Bag(size: 1)
 * %bag_3 = Bag(size: 1)
 *
 * %obj_1 = ... at %bag_2
 * %obj_2 = ... at %bag_3
 *
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0]) <- shuffle_1
 * Shuffle(from: %bag_1, into: %bag_3, [0 -> 0]) <- shuffle_2
 *
 * Note that the content of %bag_3 after shuffle_2 is identical to that of %bag_2 after shuffle_1,
 * so the following code is identical to the above one:
 *
 * %bag_1 = Bag(size: 4)
 * %bag_2 = Bag(size: 1)
 * %bag_3 = Bag(size: 1)
 *
 * %obj_1 = ... at %bag_2
 * %obj_2 = ... at %bag_2 <- HERE
 *
 * Shuffle(from: %bag_1, into: %bag_2, [0 -> 0]) <- shuffle_1
 * Shuffle(from: %bag_1, into: %bag_3, [0 -> 0]) <- shuffle_2
 *
 * "hoist_object" optimization rewrites the former code as the latter one.
 *
 * NOTE "hoist_object" DOES NOT change any instruction. It just updates the base bag of objects of
 *      interest.
 */
void hoist_object(enco::Code *code);

} // namespace enco

#endif // __ENCO_OPTIMIZATIONS_H__
