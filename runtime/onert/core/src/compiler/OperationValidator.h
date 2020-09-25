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

#ifndef __ONERT_COMPILER_OPERATION_VALIDATOR_H__
#define __ONERT_COMPILER_OPERATION_VALIDATOR_H__

#include "ir/OperationVisitor.h"

namespace onert
{
namespace ir
{
class Graph;
class Operands;
} // namespace ir
} // namespace onert

namespace onert
{
namespace compiler
{

class OperationValidator : public ir::OperationVisitor
{
public:
  OperationValidator(void) = delete;
  OperationValidator(const ir::Graph &graph);

public:
  void operator()();

public:
  void visit(const ir::operation::Comparison &node) override;
  void visit(const ir::operation::ElementwiseActivation &node) override;

private:
  // TODO Remove _ctx field
  const ir::Graph &_graph;
  const ir::Operands &_ctx;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_OPERATION_VALIDATOR_H__
