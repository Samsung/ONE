/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_TRAIN_PASS_BIAS_INSERTION_PASS_H__
#define __ONERT_COMPILER_TRAIN_PASS_BIAS_INSERTION_PASS_H__

#include "../../pass/Pass.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace compiler
{
namespace train
{
namespace pass
{

class BiasInsertionPass final : public compiler::pass::Pass, public ir::OperationVisitor
{
public:
  BiasInsertionPass(ir::Graph &graph) : compiler::pass::Pass{graph} {}

public:
  std::string id() final { return "BiasInsertionPass"; }
  void run() final;

public:
  void visit(const ir::operation::Conv2D &node) override;
  void visit(const ir::operation::DepthwiseConv2D &node) override;
  void visit(const ir::operation::FullyConnected &node) override;

private:
  ir::OperationIndex _current_op_index;
};

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_PASS_BIAS_INSERTION_PASS_H__
