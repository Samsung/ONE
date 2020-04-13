/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONNX_NODE_CONVERTER_REGISTRY_H__
#define __ONNX_NODE_CONVERTER_REGISTRY_H__

#include "onnx/onnx.pb.h"
#include "mir/Graph.h"

#include <map>
#include <string>
#include <vector>

namespace mir_onnx
{

class ModelContext
{
public:
  explicit ModelContext(const onnx::ModelProto *model);

  void setDomainOpsetVersion(const std::string &domain, const int64_t opset_version);
  int64_t getDomainOpsetVersion(const std::string &domain) const;

private:
  std::map<std::string, int64_t> _domainToOpsetVersion;
};

class ConverterContext
{
public:
  explicit ConverterContext(mir::Graph *graph);
  ~ConverterContext() = default;

  void setOutput(const std::string &name, mir::Operation::Output *output);
  mir::Operation::Output *getOutput(const std::string &name) const;
  std::vector<mir::Operation::Output *> getNodeInputs(const onnx::NodeProto &onnx_node) const;
  void setNodeOutputs(const onnx::NodeProto &onnx_node,
                      const std::vector<mir::Operation::Output *> &outputs);
  mir::Graph *getGraph() const { return _graph; }

private:
  std::map<std::string, mir::Operation::Output *> _tensorNameToOutput;
  mir::Graph *_graph;
};

class NodeConverterRegistry
{
public:
  using ConverterFunc = void (*)(const onnx::NodeProto &onnx_node, ConverterContext *context);

  NodeConverterRegistry() = default;

  ConverterFunc lookup(const std::string &optype, int64_t opset) const;
  void registerConverter(const std::string &op_type, int64_t opset, ConverterFunc conv);

  static NodeConverterRegistry &getInstance();

private:
  using VersionMap = std::map<int64_t, ConverterFunc>;

  std::unordered_map<std::string, VersionMap> _converter_map;
};

} // namespace mir_onnx

#endif // __ONNX_NODE_CONVERTER_REGISTRY_H__
