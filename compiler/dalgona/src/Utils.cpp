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
  std::vector<ptrdiff_t> shape(tensor->shape().num_dims());
  for (int i = 0; i < tensor->shape().num_dims(); i++)
  {
    shape[i] = tensor->shape().dim(i);
  }

  switch (tensor->element_type())
  {
    case loco::DataType::FLOAT32:
      return py::array_t<float, py::array::c_style>(shape, tensor->data<float>());
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

std::vector<py::dict> inputsPyArray(const luci::CircleNode *node,
                                    luci_interpreter::Interpreter *interpreter)
{
  std::vector<py::dict> inputs;
  for (uint32_t i = 0; i < node->arity(); ++i)
  {
    const auto input_tensor = interpreter->getTensor(node->arg(i));
    auto circle_node = static_cast<luci::CircleNode *>(node->arg(i));
    auto py_input = py::dict("name"_a = circle_node->name(), "data"_a = numpyArray(input_tensor),
                             "quantparam"_a = quantizationParameters(input_tensor));
    inputs.push_back(py_input);
  }
  return inputs;
}

// Note: Only returns 1 output
py::dict outputPyArray(const luci::CircleNode *node, luci_interpreter::Interpreter *interpreter)
{
  const auto output_tensor = interpreter->getTensor(node);
  auto py_output = py::dict("name"_a = node->name(), "data"_a = numpyArray(output_tensor),
                            "quantparam"_a = quantizationParameters(output_tensor));
  return py_output;
}

} // namespace dalgona
