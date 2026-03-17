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

#ifndef __CONSTANT_FOLDING_H__
#define __CONSTANT_FOLDING_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Evaluate "constant" expressions at compile time
 */
void fold_constants(enco::Code *);

struct ConstantFoldingPass final : public Pass
{
  PASS_CTOR(ConstantFoldingPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { fold_constants(code(sess)); }
};

} // namespace enco

#endif // __CONSTANT_FOLDING_H__
