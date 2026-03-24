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

#include "ONNXNodeConverterRegistry.h"

#include <memory>

namespace mir_onnx
{

void ModelContext::setDomainOpsetVersion(const std::string &domain, const int64_t opset_version)
{
  _domainToOpsetVersion.emplace(domain, opset_version);
}

int64_t ModelContext::getDomainOpsetVersion(const std::string &domain) const
{
  auto iter = _domainToOpsetVersion.find(domain);
  if (iter == _domainToOpsetVersion.end())
    throw std::runtime_error("Didn't have domain " + domain + "!");
  return iter->second;
}

ModelContext::ModelContext(const onnx::ModelProto *model)
{
  if (model == nullptr)
  {
    throw std::runtime_error{"Model should be imported before importer prepare"};
  }

  if (model->ir_version() > onnx::IR_VERSION)
  {
    throw std::runtime_error("IR version " + std::to_string(model->ir_version()) +
                             " is not supported yet.");
  }

  // Set Opset Version for each domain
  for (const auto &op_set : model->opset_import())
  {
    setDomainOpsetVersion(op_set.domain(), op_set.version());
  }
}

// ConverterContext

ConverterContext::ConverterContext(mir::Graph *graph) : _graph(graph) {}

void ConverterContext::setOutput(const std::string &name, mir::Operation::Output *output)
{
  output->setName(name);
  auto result = _tensorNameToOutput.emplace(name, output);
  if (!result.second)
    throw std::runtime_error("Name duplication: " + name);
}

mir::Operation::Output *ConverterContext::getOutput(const std::string &name) const
{
  auto iter = _tensorNameToOutput.find(name);
  if (iter == _tensorNameToOutput.end())
    return nullptr;
  else
    return iter->second;
}

std::vector<mir::Operation::Output *>
ConverterContext::getNodeInputs(const onnx::NodeProto &onnx_node) const
{
  const auto &input_names = onnx_node.input();
  std::vector<mir::Operation::Output *> outputs;

  for (const auto &input_name : input_names)
  {
    if (!input_name.empty())
    {
      auto *mir_output = getOutput(input_name);
      assert(mir_output != nullptr);
      outputs.emplace_back(mir_output);
    }
  }
  return outputs;
}

void ConverterContext::setNodeOutputs(const onnx::NodeProto &onnx_node,
                                      const std::vector<mir::Operation::Output *> &outputs)
{
  assert(!outputs.empty());
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    setOutput(onnx_node.output(i), outputs[i]);
  }
}

// NodeConverterRegistry

NodeConverterRegistry::ConverterFunc NodeConverterRegistry::lookup(const std::string &optype,
                                                                   int64_t opset) const
{
  auto it = _converter_map.find(optype);
  if (it == _converter_map.end())
  {
    return nullptr;
  }

  const VersionMap &conv_map = it->second;

  auto res = std::lower_bound(
    conv_map.crbegin(), conv_map.crend(), opset,
    [](const VersionMap::value_type &pair, int64_t opset) { return pair.first > opset; });

  if (res == conv_map.crend())
  {
    return nullptr;
  }
  return res->second;
}

NodeConverterRegistry &NodeConverterRegistry::getInstance()
{
  static NodeConverterRegistry instance;
  return instance;
}

void NodeConverterRegistry::registerConverter(const std::string &op_type, int64_t opset,
                                              NodeConverterRegistry::ConverterFunc conv)
{
  _converter_map[op_type].emplace(opset, conv);
}

} // namespace mir_onnx
