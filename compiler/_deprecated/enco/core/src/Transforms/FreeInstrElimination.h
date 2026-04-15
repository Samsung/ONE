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

#ifndef __ENCO_TRANSFORM_FREE_INSTR_ELIMINATION_H__
#define __ENCO_TRANSFORM_FREE_INSTR_ELIMINATION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Eliminate free instructions
 *
 * An instruction is referred to as "free" if it is not bound to any "block"
 */
void eliminate_free_instr(coco::Module *mod);

/**
 * @brief Eliminate free instructions
 */
static inline void eliminate_free_instr(enco::Code *code)
{
  // This function is just a wrapper of the above "void eliminate_free_instr(coco::Module *mod)"
  eliminate_free_instr(code->module());
}

struct FreeInstrEliminationPass final : public Pass
{
  PASS_CTOR(FreeInstrEliminationPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { eliminate_free_instr(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_FREE_INSTR_ELIMINATION_H__
