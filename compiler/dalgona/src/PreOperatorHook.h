/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DALGONA_PRE_OPERATOR_HOOK_H__
#define __DALGONA_PRE_OPERATOR_HOOK_H__

#include "Utils.h"
#include "StringUtils.h"

#include <loco/IR/Node.h>
#include <luci_interpreter/Interpreter.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <pybind11/embed.h>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

namespace dalgona
{

// Invoke a user-written Python hook before an operator is executed
class PreOperatorHook final : public luci::CircleNodeVisitor<void>
{
private:
  py::object _analysis;
  luci_interpreter::Interpreter *_interpreter{nullptr};

public:
  explicit PreOperatorHook(py::object analysis, luci_interpreter::Interpreter *interpreter)
    : _analysis(analysis), _interpreter(interpreter)
  {
    // Do nothing
  }

  // default
  void visit(const luci::CircleNode *node)
  {
    if (not py::hasattr(_analysis, "DefaultOpPre"))
      return;

    py::object hook = _analysis.attr("DefaultOpPre");
    auto inputs = inputsPyArray(node, _interpreter);

    py::list input_list;
    for (int i = 0; i < inputs.size(); i++)
    {
      input_list.append(inputs[i]);
    }

    pySafeCall(hook,
               node->name(),             // name
               toString(node->opcode()), // opcode
               input_list                // list of inputs
    );
  }
};

} // namespace dalgona

#endif // __DALGONA_PRE_OPERATOR_HOOK_H__
