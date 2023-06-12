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

#ifndef __ONERT_IR_PASS_LOWERED_OPERAND_PASS_H__
#define __ONERT_IR_PASS_LOWERED_OPERAND_PASS_H__

#include "OperandPass.h"
#include "compiler/ILoweredGraph.h"

namespace onert
{
namespace compiler
{
namespace pass
{

class LoweredOperandPass : public OperandPass
{
public:
  LoweredOperandPass(compiler::ILoweredGraph &lowered_graph)
    : OperandPass{lowered_graph.graph()}, _lowered_graph{lowered_graph}
  {
    // DO NOTHING
  }

  virtual ~LoweredOperandPass() = default;

  std::string id() override = 0;
  void callback(const ir::OperandIndex &i, ir::Operand &o) override = 0;

protected:
  compiler::ILoweredGraph &_lowered_graph;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_IR_PASS_LOWERED_OPERAND_PASS_H__
