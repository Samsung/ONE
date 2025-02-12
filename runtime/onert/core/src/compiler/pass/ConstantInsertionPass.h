/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_PASS_CONSTANT_INSERTION_PASS_H__
#define __ONERT_COMPILER_PASS_CONSTANT_INSERTION_PASS_H__

#include "LoweredOperationPass.h"
#include "backend/Backend.h"
#include "ir/Index.h"

#include <unordered_map>

namespace onert::compiler::pass
{

class ConstantInsertionPass : public LoweredOperationPass
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "ConstantInsertionPass"; }

public:
  void callback(const ir::OperationIndex &index, ir::IOperation &node) final;

private:
  std::unordered_map<const backend::Backend *, ir::OperandIndex> _replace_operands_map;
  std::unordered_map<ir::OperandIndex, const backend::Backend *> _keep_operands_map;
};

} // namespace onert::compiler::pass

#endif // __ONERT_COMPILER_PASS_CONSTANT_INSERTION_PASS_H__
