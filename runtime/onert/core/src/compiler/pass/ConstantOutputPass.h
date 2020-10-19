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

#ifndef __ONERT_COMPILER_PASS_CONSTANT_OUTPUT_PASS_H__
#define __ONERT_COMPILER_PASS_CONSTANT_OUTPUT_PASS_H__

#include "OperandPass.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief Pass to specially handle constant model outputs
 *
 * As an output buffer is given right before an execution but constant initialization is done at
 * prepare phase, the current runtime structure cannot handle when an output is constant.
 * To resolve this problem, this pass inserts a Permute layer with a const input and make the model
 * output tensor to be its output.
 *
 * e.g.)
 *
 * ((Const Output))
 *
 * becomes
 *
 * (Const) -> [Permute] -> ((Output))
 *
 * Note that this is a mandatory pass for Graph.
 */
class ConstantOutputPass : public OperandPass
{
public:
  using OperandPass::OperandPass;

public:
  std::string id() final { return "ConstantOutputPass"; }

public:
  void callback(const ir::OperandIndex &i, ir::Operand &o) final;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_CONSTANT_INSERTION_PASS_H__
