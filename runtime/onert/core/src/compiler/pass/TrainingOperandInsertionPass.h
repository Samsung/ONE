/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_TRAINING_OPERAND_INSERTION_PASS_H__
#define __ONERT_COMPILER_TRAINING_OPERAND_INSERTION_PASS_H__

#include "ir/OperationVisitor.h"
#include "OperationPass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

class TrainingOperandInsertionPass : public OperationPass, public ir::MutableOperationVisitor
{
public:
  using OperationPass::OperationPass;

public:
  std::string id() final { return "TrainingOperandInsertionPass"; }

public:
  void callback(const ir::OperationIndex &i, ir::Operation &n) final;

private:
  void visit(ir::operation::ElementwiseActivation &) final;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAINING_OPERAND_INSERTION_PASS_H__
