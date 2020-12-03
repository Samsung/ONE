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

#include "Interpreter.h"

#include <stack>
#include <unordered_set>

#include "Registration.h"

#include "ir/OperandIndexMap.h"
#include "util/logging.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace interp
{

// TODO more structured execution kernel implementation
// TODO use cker for execution
// TODO divide tensor prepare and execution
// TODO introduce memory manager (buffer allocate and free)
class OperationExecutor
{
public:
  OperationExecutor(ExecEnv *env) : _env{env}
  {
#define INTERP_OP(InternalName) _kernels[ir::OpCode::InternalName] = get##InternalName();
#include "InterpOps.lst"
#undef INTERP_OP
  }

  void execute(const ir::OperationIndex &idx)
  {
    const ir::Operation &node = _env->graph().operations().at(idx);
    const auto nodeName = node.name();
    VERBOSE(INTERPRETER) << "Prepare output operands and execute " << nodeName
                         << " operation (id: " << idx << ")" << std::endl;

    const auto nodeOpCode = node.opcode();
    if (_kernels.find(nodeOpCode) == _kernels.end())
    {
      throw std::runtime_error{"Interpreter: Operation " + nodeName + " is not yet implemented"};
    }

    if (_kernels[nodeOpCode]->prepare != nullptr)
    {
      _kernels[nodeOpCode]->prepare(_env, node);
    }
    _kernels[nodeOpCode]->invoke(_env, node);
  }

private:
  ExecEnv *_env;
  std::unordered_map<ir::OpCode, OpKernel *> _kernels;
};

void Interpreter::run()
{
  VERBOSE(INTERPRETER) << "Interpreter is invoked " << std::endl;

  // operand_stack: save operands prepared to use
  std::stack<ir::OperandIndex> operand_stack;

  // Note: We should push input first, then constant.
  //       We use use-def for find operators ready to execution,
  //       but Use-Def cannot handle parameters (maybe constant, but not always)
  // Note: If all model inputs are constant, it may not work (depend on tensors' order).
  //       But that scenario may not exist
  for (auto ind : _env->graph().getInputs())
  {
    VERBOSE(INTERPRETER) << "Input: Push to operand stack " << ind << std::endl;

    operand_stack.push(ind);
  }

  _env->graph().operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (obj.isConstant())
    {
      VERBOSE(INTERPRETER) << "Constant: Push to operand stack " << ind << std::endl;

      operand_stack.push(ind);
    }
  });

  // Execution
  std::unordered_set<ir::OperandIndex> ready_check;
  std::unordered_set<ir::OperationIndex> executed;
  OperationExecutor executor{_env.get()};
  while (!operand_stack.empty())
  {
    const auto current_operand_index = operand_stack.top();
    operand_stack.pop();
    VERBOSE(INTERPRETER) << "Poped operand " << current_operand_index.value()
                         << " is checked ready to use" << std::endl;

    assert(ready_check.find(current_operand_index) == ready_check.end());
    ready_check.insert(current_operand_index);

    // Find prepared operations by scan use of current operand
    std::stack<ir::OperationIndex> operation_stack;
    const auto use_operators = _env->graph().operands().at(current_operand_index).getUses();
    for (const auto &use_operator : use_operators)
    {
      // Assumption: all parameters are ready to use
      bool operator_ready = true;
      for (auto input_index : _env->graph().operations().at(use_operator).getInputs())
      {
        if (ready_check.find(input_index) == ready_check.end())
        {
          operator_ready = false;
          break;
        }
      }

      if (operator_ready)
      {
        VERBOSE(INTERPRETER) << "Ready to execute operation " << use_operator << std::endl;
        operation_stack.push(use_operator);
      }
    }

    while (!operation_stack.empty())
    {
      const auto current_operation_index = operation_stack.top();
      operation_stack.pop();
      VERBOSE(INTERPRETER) << "Poped operation: " << current_operation_index << "("
                           << _env->graph().operations().at(current_operation_index).name() << ")"
                           << std::endl;

      // execution
      // 1. Prepare output tensor
      // 2. Call operation kernel
      executor.execute(current_operation_index);
      executed.insert(current_operation_index);

      // 3. Push each output into operand stack
      const auto def_operands = _env->graph().operations().at(current_operation_index).getOutputs();
      for (auto def_operand : def_operands)
      {
        VERBOSE(INTERPRETER) << "Buffer: Push to operand stack " << def_operand.value()
                             << std::endl;
        operand_stack.push(def_operand);
      }

      // 4. Free if lifetime of buffer operands used by input is finished
      for (auto input_index : _env->graph().operations().at(current_operation_index).getInputs())
      {
        const auto use_operators = _env->graph().operands().at(input_index).getUses();
        bool dead_buffer = true;
        for (const auto &use_operator : use_operators)
        {
          if (executed.find(use_operator) == executed.end())
          {
            dead_buffer = false;
            break;
          }
        }

        if (dead_buffer)
        {
          _env->freeIfAllocated(input_index);
        }
      }
    }
  }
}

} // namespace interp
} // namespace onert
