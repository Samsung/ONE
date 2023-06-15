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

#ifndef __ONERT_COMPILER_PASS_PERMUTATION_OPERATION_PASS_H__
#define __ONERT_COMPILER_PASS_PERMUTATION_OPERATION_PASS_H__

#include "ir/OperationVisitor.h"
#include "LoweredOperationPass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

class PermutationOperationPass : public LoweredOperationPass, public ir::OperationVisitor
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "PermutationOperationPass"; }

public:
  void callback(const ir::OperationIndex &i, ir::IOperation &n) final;

public:
  void visit(const ir::operation::BinaryArithmetic &) final;
  void visit(const ir::operation::Comparison &) final;
  void visit(const ir::operation::Concat &) final;
  void visit(const ir::operation::ElementwiseBinary &) final;
  void visit(const ir::operation::ElementwiseUnary &) final;
  void visit(const ir::operation::OneHot &) final;
  void visit(const ir::operation::Pack &) final;
  void visit(const ir::operation::PReLU &) final;
  void visit(const ir::operation::SquaredDifference &) final;
  void visit(const ir::operation::Unpack &) final;
  void visit(const ir::operation::FullyConnected &) final;
  void visit(const ir::operation::Gather &) final;
  void visit(const ir::operation::Reshape &) final;

private:
  void applyExpandRanks(const ir::Operation &);
  void changeToKeepLayout(const ir::Operation &);
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_PERMUTATION_OPERATION_PASS_H__
