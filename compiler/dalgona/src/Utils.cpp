/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <pybind11/numpy.h>
#include <stdexcept>

using Tensor = luci_interpreter::Tensor;

namespace py = pybind11;
using namespace py::literals;

namespace
{

py::array numpyArray(const Tensor *tensor)
{
  uint32_t size = 1;
  std::vector<ptrdiff_t> shape(tensor->shape().num_dims());
  for (int i = 0; i < tensor->shape().num_dims(); i++)
  {
    shape[i] = tensor->shape().dim(i);
    size *= shape[i];
  }

  if (size == 0)
    return dalgona::none();

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

py::dict quantizationParameters(const Tensor *tensor)
{
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

py::object none()
{
  // return py::cast<py::none>(Py_None);
  return py::none();
}

std::vector<py::dict> inputsPyArray(const luci::CircleNode *node,
                                    luci_interpreter::Interpreter *interpreter)
{
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
               "quantparam"_a = quantizationParameters(input_tensor),
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

    // skip invalid outputs (e.g., non-existing bias in TCONV)
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
      continue;

    auto py_output =
      py::dict("name"_a = circle_node->name(), "data"_a = numpyArray(output_tensor),
               "quantparam"_a = quantizationParameters(output_tensor),
               "is_const"_a = circle_node->opcode() == luci::CircleOpcode::CIRCLECONST);
    outputs.push_back(py_output);
  }
  return outputs;
}

// Note: Only returns 1 output
py::dict outputPyArray(const luci::CircleNode *node, luci_interpreter::Interpreter *interpreter)
{
  const auto output_tensor = interpreter->getTensor(node);
  auto py_output = py::dict("name"_a = node->name(), "data"_a = numpyArray(output_tensor),
                            "quantparam"_a = quantizationParameters(output_tensor),
                            "is_const"_a = node->opcode() == luci::CircleOpcode::CIRCLECONST);
  return py_output;
}

const std::string toString(luci::CircleOpcode opcode)
{
  static const char *names[] = {
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
#undef CIRCLE_VNODE
  };
  // Returned string is the substring of circle class name ("Circle" is sliced out)
  return std::string(names[static_cast<int>(opcode)]).substr(6);
}

const std::string toString(luci::FusedActFunc fused_act)
{
  switch (fused_act)
  {
    case (luci::FusedActFunc::UNDEFINED):
      return std::string("undefined");
    case (luci::FusedActFunc::NONE):
      return std::string("none");
    case (luci::FusedActFunc::RELU):
      return std::string("relu");
    case (luci::FusedActFunc::RELU_N1_TO_1):
      return std::string("relu_n1_to_1");
    case (luci::FusedActFunc::RELU6):
      return std::string("relu6");
    default:
      throw std::runtime_error("Unsupported activation function");
  }
}

} // namespace dalgona
