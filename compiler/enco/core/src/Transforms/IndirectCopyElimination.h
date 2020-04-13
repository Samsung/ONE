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

#ifndef __ENCO_TRANSFORM_INDIRECT_COPY_ELIMINATION_H__
#define __ENCO_TRANSFORM_INDIRECT_COPY_ELIMINATION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Convert all the indirect copies as a direct copy
 *
 * >>> BEFORE <<<
 * %obj_0 = ...
 * %obj_1 = ...
 * %obj_2 = ...
 *
 * copy(from: %obj_0, into: %obj_1)
 * copy(from: %obj_1, into: %obj_2)
 *
 * >>> AFTER <<<
 * %obj_0 = ...
 * %obj_1 = ...
 * %obj_2 = ...
 *
 * copy(from: %obj_0, into: %obj_1)
 * copy(from: %obj_0, into: %obj_2)
 *
 */
void eliminate_indirect_copy(enco::Code *code);

struct IndirectCopyEliminationPass final : public enco::Pass
{
  PASS_CTOR(IndirectCopyEliminationPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { eliminate_indirect_copy(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_INDIRECT_COPY_ELIMINATION_H__
