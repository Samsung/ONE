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

#ifndef __ONERT_COMPILER_TRAIN_PASS_TRAINABLE_CONSTANT_INSERTION_PASS_H__
#define __ONERT_COMPILER_TRAIN_PASS_TRAINABLE_CONSTANT_INSERTION_PASS_H__

#include "../../pass/LoweredOperationPass.h"

namespace onert::compiler::train::pass
{

// TODO Consider to insert trainable constants only when the correspoding constant is a training
//      parameter(weights or bias) if there are memory issues.
class TrainableConstantInsertionPass : public compiler::pass::LoweredOperationPass
{
public:
  using LoweredOperationPass::LoweredOperationPass;

public:
  std::string id() final { return "TrainableConstantInsertionPass"; }

public:
  void callback(const ir::OperationIndex &index, ir::IOperation &node) final;

private:
  ir::OperandIndex insertNewOperand(const ir::Operand &object);
  void updateUseDef(const ir::OperandIndex &old_index, const ir::OperandIndex &new_index,
                    const ir::OperationIndex &node_index);
};

} // namespace onert::compiler::train::pass

#endif // __ONERT_COMPILER_TRAIN_PASS_TRAINABLE_CONSTANT_INSERTION_PASS_H__
