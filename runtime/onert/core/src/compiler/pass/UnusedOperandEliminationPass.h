/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  UnusedOperandEliminationPass.h
 * @brief This file contains UnusedOperandEliminationPass class
 */

#ifndef __ONERT_COMPILER_PASS_UNUSED_OPERAND_ELIMINATION_PASS_H__
#define __ONERT_COMPILER_PASS_UNUSED_OPERAND_ELIMINATION_PASS_H__

#include "Pass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief  A pass to eliminate unused operands from the graph
 *
 * Remove operands that are not used by any operations, except Graph inputs/outputs.
 *
 */
class UnusedOperandEliminationPass : public Pass
{
public:
  using Pass::Pass;

public:
  std::string id() override { return "UnusedOperandEliminationPass"; }
  void run() final;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_UNUSED_OPERAND_ELIMINATION_PASS_H__
