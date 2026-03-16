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

#ifndef __INTRINSIC_SELECTION_H__
#define __INTRINSIC_SELECTION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Select Intricsic (API) to be used
 *
 * This pass is analogue of "Instruction Selection" pass. This "Intrisic Selection" pass
 * will replace a general coco IR instruction into a backend-specific coco (extended) IR
 * instruction.
 */
void select_intrinsic(enco::Code *);

struct IntrinsicSelectionPass final : public Pass
{
  PASS_CTOR(IntrinsicSelectionPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { select_intrinsic(code(sess)); }
};

} // namespace enco

#endif // __INTRINSIC_SELECTION_H__
