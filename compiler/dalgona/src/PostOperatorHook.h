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

#ifndef __DALGONA_POST_OPERATOR_HOOK_H__
#define __DALGONA_POST_OPERATOR_HOOK_H__

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

// Invoke a user-written Python hook after an operator is executed
class PostOperatorHook final : public luci::CircleNodeVisitor<void>
{

// This macro creates three variables used for post-operator hooks.
// 1. hook: Python function to be invoked (type: py::object)
// 2. inputs: input data (type: std::vector of numpy array)
// 3. output: output data (type: numpy array)
#define POST_OPERATOR_HOOK_PROLOGUE(OP_NAME)                \
  if (!py::hasattr(_analysis, #OP_NAME "Post"))             \
  {                                                         \
    visit(loco::must_cast<const luci::CircleNode *>(node)); \
    return;                                                 \
  }                                                         \
  py::object hook = _analysis.attr(#OP_NAME "Post");        \
  auto inputs = inputsPyArray(node, _interpreter);          \
  auto output = outputPyArray(node, _interpreter);

// Multi-output version of POST_OPERATOR_HOOK_PROLOGUE
#define POST_OPERATOR_HOOK_PROLOGUE_MULTI_OUTS(OP_NAME)     \
  if (!py::hasattr(_analysis, #OP_NAME "Post"))             \
  {                                                         \
    visit(loco::must_cast<const luci::CircleNode *>(node)); \
    return;                                                 \
  }                                                         \
  py::object hook = _analysis.attr(#OP_NAME "Post");        \
  auto inputs = inputsPyArray(node, _interpreter);          \
  auto outputs = outputsPyArray(node, _interpreter);

private:
  py::object _analysis;
  luci_interpreter::Interpreter *_interpreter{nullptr};

public:
  explicit PostOperatorHook(py::object analysis, luci_interpreter::Interpreter *interpreter)
    : _analysis(analysis), _interpreter(interpreter)
  {
    // Do nothing
  }

  // default
  void visit(const luci::CircleNode *node)
  {
    if (not py::hasattr(_analysis, "DefaultOpPost"))
      return;

    if (isMultiOutNode(node))
      return;

    py::object hook = _analysis.attr("DefaultOpPost");
    auto inputs = inputsPyArray(node, _interpreter);
    auto output = outputPyArray(node, _interpreter);

    py::list input_list;
    for (uint32_t i = 0; i < inputs.size(); i++)
    {
      input_list.append(inputs[i]);
    }

    pySafeCall(hook,
               node->name(),             // name
               toString(node->opcode()), // opcode
               input_list,               // list of inputs
               output                    // output
    );
  }

  void visit(const luci::CircleConv2D *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(Conv2D)

    auto padding = node->padding();
    auto stride = node->stride();
    auto dilation = node->dilation();

    auto py_stride = py::dict("w"_a = stride->w(), "h"_a = stride->h());
    auto py_dilation = py::dict("w"_a = dilation->w(), "h"_a = dilation->h());

    auto fused_act = node->fusedActivationFunction();

    pySafeCall(hook,
               node->name(),                                      // name
               inputs[0],                                         // input
               inputs[1],                                         // filter
               inputs[2],                                         // bias
               padding == luci::Padding::SAME ? "SAME" : "VALID", // padding
               py_stride,                                         // stride
               py_dilation,                                       // dilation
               output,                                            // output
               toString(fused_act)                                // fused activation
    );
  }

  void visit(const luci::CircleDepthwiseConv2D *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(DepthwiseConv2D)

    auto padding = node->padding();
    auto stride = node->stride();
    auto dilation = node->dilation();
    auto depthMultiplier = node->depthMultiplier();

    auto py_stride = py::dict("w"_a = stride->w(), "h"_a = stride->h());
    auto py_dilation = py::dict("w"_a = dilation->w(), "h"_a = dilation->h());

    auto fused_act = node->fusedActivationFunction();

    pySafeCall(hook,
               node->name(),                                      // name
               inputs[0],                                         // input
               inputs[1],                                         // filter
               inputs[2],                                         // bias
               padding == luci::Padding::SAME ? "SAME" : "VALID", // padding
               py_stride,                                         // stride
               depthMultiplier,                                   // depthMultiplier
               py_dilation,                                       // dilation
               output,                                            // output
               toString(fused_act)                                // fused activation
    );
  }

  void visit(const luci::CircleAdd *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(Add)

    auto fused_act = node->fusedActivationFunction();

    pySafeCall(hook,
               node->name(),       // name
               inputs[0],          // x
               inputs[1],          // y
               output,             // output
               toString(fused_act) // fused activation
    );
  }

  void visit(const luci::CircleFullyConnected *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(FullyConnected)

    auto fused_act = node->fusedActivationFunction();

    pySafeCall(hook,
               node->name(),       // name
               inputs[0],          // input
               inputs[1],          // weights
               inputs[2],          // bias
               output,             // output
               toString(fused_act) // fused activation
    );
  }

  void visit(const luci::CircleTransposeConv *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(TransposeConv)

    auto padding = node->padding();
    auto stride = node->stride();

    auto py_stride = py::dict("w"_a = stride->w(), "h"_a = stride->h());

    pySafeCall(hook,
               node->name(),                                      // name
               inputs[2],                                         // input
               inputs[1],                                         // filter
               inputs[0],                                         // output shape
               inputs.size() == 4 ? inputs[3] : none(),           // bias
               padding == luci::Padding::SAME ? "SAME" : "VALID", // padding
               py_stride,                                         // stride
               output                                             // output
    );
  }

  void visit(const luci::CircleInstanceNorm *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE(InstanceNorm)

    auto epsilon = node->epsilon();

    auto fused_act = node->fusedActivationFunction();

    pySafeCall(hook,
               node->name(),       // name
               inputs[0],          // input
               inputs[1],          // gamma
               inputs[2],          // beta
               epsilon,            // epsilon
               output,             // output
               toString(fused_act) // fused activation
    );
  }

  void visit(const luci::CircleSplit *node)
  {
    POST_OPERATOR_HOOK_PROLOGUE_MULTI_OUTS(Split)

    py::list output_list;
    for (uint32_t i = 0; i < outputs.size(); i++)
    {
      output_list.append(outputs[i]);
    }

    auto num_split = node->num_split();

    pySafeCall(hook,
               node->name(), // name
               inputs[0],    // split_dim
               inputs[1],    // input
               num_split,    // num_split
               output_list   // list of outputs
    );
  }

#undef POST_OPERATOR_HOOK_PROLOGUE_MULTI_OUTS
};

} // namespace dalgona

#endif // __DALGONA_POST_OPERATOR_HOOK_H__
