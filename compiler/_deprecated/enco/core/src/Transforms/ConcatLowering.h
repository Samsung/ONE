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

#ifndef __ENCO_CONCAT_LOWERING_H__
#define __ENCO_CONCAT_LOWERING_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Lower eval(Concat(...)) as a sequence of shuffle instructions
 */
void lower_concat(enco::Code *code);

struct ConcatLoweringPass final : public Pass
{
  PASS_CTOR(ConcatLoweringPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { lower_concat(code(sess)); }
};

} // namespace enco

#endif // __ENCO_CONCAT_LOWERING_H__
