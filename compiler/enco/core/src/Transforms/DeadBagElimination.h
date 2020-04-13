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

#ifndef __ENCO_TRANSFORM_DEAD_BAG_ELIMINATION_H__
#define __ENCO_TRANSFORM_DEAD_BAG_ELIMINATION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Eliminate dead bags
 *
 * A bag is referred to as dead if it is neither input nor output, and has no read. If a bag is
 * dead, it is unnecessary to updates its values as these values are never used.
 *
 * "eliminate_dead_bag" removes all the dead bags and its updaters from IR.
 */
void eliminate_dead_bag(enco::Code *code);

struct DeadBagEliminationPass final : public Pass
{
  PASS_CTOR(DeadBagEliminationPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { eliminate_dead_bag(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_DEAD_BAG_ELIMINATION_H__
