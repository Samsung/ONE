/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "util/ShapeInference.h"
#include <map>

namespace onert
{
namespace shape_inference
{

std::vector<std::string> strSplit(const std::string &text, const std::string delimiter)
{
  std::vector<std::string> result;

  size_t start = 0;
  size_t pos = 0;

  do
  {
    pos = text.find(delimiter, start);
    if (pos == std::string::npos)
    {
      result.push_back(text.substr(start, text.size() - start));
      break;
    }

    result.push_back(text.substr(start, pos - start));
    start = pos + delimiter.size();
  } while (pos != std::string::npos);

  return result;
}

void parseEinsumEquation(const std::string &equation, std::vector<std::string> &input_subscripts,
                         std::string &output_subscript)
{
  std::vector<std::string> inputs_and_output_subscripts = strSplit(equation, "->");
  if (inputs_and_output_subscripts.size() != 2)
  {
    throw std::runtime_error{"Einsum: Expecting exactly one '->' in einsum equation: " + equation};
  }

  output_subscript = inputs_and_output_subscripts[1];
  input_subscripts = strSplit(inputs_and_output_subscripts[0], ",");
  if (input_subscripts.size() != 1 && input_subscripts.size() != 2)
  {
    throw std::runtime_error{"Einsum: Expecting 1 or 2 input subscripts in equation '" + equation +
                             "' but got: " + std::to_string(input_subscripts.size())};
  }
}

inline ir::Shape inferEinsumShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape,
                                  const ir::operation::Einsum::Param &param)
{
  // TODO: increase coverage
  // Here, we only cover easy equations e.g. abc,cde->abde

  std::vector<std::string> input_str;
  std::string output_str;

  parseEinsumEquation(param.equation, input_str, output_str);

  // For input_str[0] and input_str[1], construct a map (char to dim value)
  std::map<char, int> char_map;
  for (size_t i = 0; i < input_str[0].size(); ++i)
  {
    char_map.insert({input_str[0].at(i), lhs_shape.dim(i)});
  }
  for (size_t i = 0; i < input_str[1].size(); ++i)
  {
    char_map.insert({input_str[1].at(i), rhs_shape.dim(i)});
  }

  ir::Shape output_shape;
  for (size_t i = 0; i < output_str.size(); ++i)
  {
    auto dim = char_map.find(output_str.at(i));
    output_shape.append(dim->second);
  }

  return output_shape;
}

void StaticInferer::visit(const ir::operation::Einsum &op)
{
  const auto lhs_index = op.getInputs().at(0);
  const auto rhs_index = op.getInputs().at(1);
  const auto output_index = op.getOutputs().at(0);
  const auto lhs = _operands.at(lhs_index);
  const auto rhs = _operands.at(rhs_index);
  auto &output = _operands.at(output_index);

  if (lhs.info().isDynamic() || rhs.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  auto new_shape = inferEinsumShape(lhs.shape(), rhs.shape(), op.param());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Einsum &op)
{
  const auto lhs_index = op.getInputs().at(0);
  const auto rhs_index = op.getInputs().at(1);
  auto lhs = _tensor_registry->getITensor(lhs_index);
  auto rhs = _tensor_registry->getITensor(rhs_index);

  if (!lhs->is_dynamic() && !rhs->is_dynamic())
    return;

  const auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  auto lhs_shape = lhs->getShape();
  auto rhs_shape = rhs->getShape();

  auto new_shape = inferEinsumShape(lhs_shape, rhs_shape, op.param());
  _dynamic_tensor_manager->applyShape(output_index, new_shape);
}

} // namespace shape_inference
} // namespace onert
