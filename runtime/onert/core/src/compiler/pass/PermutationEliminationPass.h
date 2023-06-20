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

#ifndef __ONERT_COMPILER_PASS_PERMUTATION_ELIMINATION_PASS_H__
#define __ONERT_COMPILER_PASS_PERMUTATION_ELIMINATION_PASS_H__

#include "ir/OperationVisitor.h"
#include "LoweredOperationPass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief An optimization pass that removes Permute operations if possible
 *
 * There may be some Permute operations that are inserted by PermutationInsertionPass or other
 * passes. This pass checks all Permute operations and eliminates them if Permute in/out tensors
 * are compatible and layouts match.
 *
 * Permute input tensor is kept and the output is removed for all the cases, except model outputs.
 * As all output tensors have to be builtin backend, so the output is kept.
 *
 * @note This is an optimization pass which means that everything should work fine even if this pass
 *       was skipped.
 */
class PermutationEliminationPass : public LoweredOperationPass, public ir::OperationVisitor
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "PermutationEliminationPass"; }

public:
  void callback(const ir::OperationIndex &i, ir::IOperation &n) final;

private:
  void visit(const ir::operation::Permute &) final;

private:
  ir::OperationIndex _op_ind;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_PERMUTATION_ELIMINATION_PASS_H__
