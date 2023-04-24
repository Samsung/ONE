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

#include "Utils.h"
#include "StringUtils.h"

#include <luci_interpreter/core/Tensor.h>
#include <luci/IR/CircleOpcode.h>
#include <luci/IR/CircleNodeDecl.h>

#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

using Tensor = luci_interpreter::Tensor;

namespace py = pybind11;
using namespace py::literals;

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

namespace
{

py::array numpyArray(const Tensor *tensor)
{
  assert(tensor != nullptr); // FIX_CALLER_UNLESS

  const auto tensor_shape = tensor->shape();

  uint32_t size = 1;
  std::vector<uint32_t> shape(tensor_shape.num_dims());
  for (int i = 0; i < tensor_shape.num_dims(); i++)
  {
    THROW_UNLESS(tensor_shape.dim(i) >= 0, "Negative dimension detected in " + tensor->name());

    shape[i] = tensor_shape.dim(i);
    size *= shape[i];
  }

  if (size == 0)
    return py::none();

  switch (tensor->element_type())
  {
    case loco::DataType::FLOAT32:
      return py::array_t<float, py::array::c_style>(shape, tensor->data<float>());
    case loco::DataType::S16:
      return py::array_t<int16_t, py::array::c_style>(shape, tensor->data<int16_t>());
    case loco::DataType::S32:
      return py::array_t<int32_t, py::array::c_style>(shape, tensor->data<int32_t>());
    case loco::DataType::S64:
      return py::array_t<int64_t, py::array::c_style>(shape, tensor->data<int64_t>());
    case loco::DataType::U8:
      return py::array_t<uint8_t, py::array::c_style>(shape, tensor->data<uint8_t>());
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

py::dict quantparam(const Tensor *tensor)
{
  assert(tensor != nullptr); // FIX_CALLER_UNLESS

  auto scale = tensor->scales();
  auto zp = tensor->zero_points();

  py::list py_scale;
  for (auto s : scale)
  {
    py_scale.append(s);
  }

  py::list py_zp;
  for (auto z : zp)
  {
    py_zp.append(z);
  }

  auto quantparam = py::dict("scale"_a = py_scale, "zero_point"_a = py_zp,
                             "quantized_dimension"_a = tensor->quantized_dimension());
  return quantparam;
}

} // namespace

namespace dalgona
{

py::object none() { return py::none(); }

std::vector<py::dict> inputsPyArray(const luci::CircleNode *node,
                                    luci_interpreter::Interpreter *interpreter)
{
  assert(node != nullptr);        // FIX_CALLER_UNLESS
  assert(interpreter != nullptr); // FIX_CALLER_UNLESS

  std::vector<py::dict> inputs;
  for (uint32_t i = 0; i < node->arity(); ++i)
  {
    const auto input_tensor = interpreter->getTensor(node->arg(i));
    auto circle_node = static_cast<luci::CircleNode *>(node->arg(i));

    // skip invalid inputs (e.g., non-existing bias in TCONV)
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
      continue;

    auto py_input =
      py::dict("name"_a = circle_node->name(), "data"_a = numpyArray(input_tensor),
               "quantparam"_a = quantparam(input_tensor),
               "is_const"_a = circle_node->opcode() == luci::CircleOpcode::CIRCLECONST);
    inputs.push_back(py_input);
  }
  return inputs;
}

std::vector<py::dict> outputsPyArray(const luci::CircleNode *node,
                                     luci_interpreter::Interpreter *interpreter)
{
  std::vector<py::dict> outputs;
  for (auto succ : loco::succs(node))
  {
    const auto output_tensor = interpreter->getTensor(succ);
    auto circle_node = static_cast<luci::CircleNode *>(succ);

    auto opcode_str = toString(circle_node->opcode());
    // Check if node is a multi-output node
    // Assumption: Multi-output virtual nodes have 'Out' prefix
    // TODO Fix this if the assumption changes
    THROW_UNLESS(opcode_str.substr(opcode_str.length() - 3) == "Out",
                 "Invalid output detected in " + node->name());

    auto py_output =
      py::dict("name"_a = circle_node->name(), "data"_a = numpyArray(output_tensor),
               "quantparam"_a = quantparam(output_tensor),
               "is_const"_a = circle_node->opcode() == luci::CircleOpcode::CIRCLECONST);
    outputs.push_back(py_output);
  }
  return outputs;
}

// Note: Only returns 1 output
py::dict outputPyArray(const luci::CircleNode *node, luci_interpreter::Interpreter *interpreter)
{
  assert(node != nullptr);        // FIX_CALLER_UNLESS
  assert(interpreter != nullptr); // FIX_CALLER_UNLESS

  const auto tensor = interpreter->getTensor(node);

  THROW_UNLESS(tensor != nullptr, "Null tensor detected in " + node->name());

  auto py_output = py::dict("name"_a = node->name(), "data"_a = numpyArray(tensor),
                            "quantparam"_a = quantparam(tensor),
                            "is_const"_a = node->opcode() == luci::CircleOpcode::CIRCLECONST);
  return py_output;
}

bool isMultiOutNode(const luci::CircleNode *node)
{
  assert(node != nullptr); // FIX_CALLER_UNLESS

  switch (node->opcode())
  {
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::SPLIT_V:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::UNPACK:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::TOPK_V2:
    case luci::CircleOpcode::CIRCLETOPKV2OUT:
      return true;
    default:
      return false;
  }
}

} // namespace dalgona

#undef THROW_UNLESS
