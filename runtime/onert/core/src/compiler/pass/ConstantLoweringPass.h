/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_PASS_CONSTANT_LOWERING_PASS_H__
#define __ONERT_COMPILER_PASS_CONSTANT_LOWERING_PASS_H__

#include <ir/Index.h>
#include "LoweredOperationPass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

class ConstantLoweringPass : public LoweredOperationPass
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "ConstantLoweringPass"; }

public:
  void callback(const ir::OperationIndex &index, ir::IOperation &node) final;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_CONSTANT_LOWERING_PASS_H__
