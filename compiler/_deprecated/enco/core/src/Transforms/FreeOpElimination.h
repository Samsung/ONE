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

#ifndef __ENCO_TRANSFORM_FREE_OP_ELIMINATION_H__
#define __ENCO_TRANSFORM_FREE_OP_ELIMINATION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Eliminate free op
 *
 * An op is referred to as "free" if it is not bound to any "instruction"
 */
void eliminate_free_op(coco::Module *mod);

/**
 * @brief Eliminate free op
 */
static inline void eliminate_free_op(enco::Code *code)
{
  // This function is just a wrapper of the above "void eliminate_free_op(coco::Module *mod)"
  eliminate_free_op(code->module());
}

struct FreeOpEliminationPass final : public Pass
{
  PASS_CTOR(FreeOpEliminationPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { eliminate_free_op(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_FREE_OP_ELIMINATION_H__
