/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "graph.h"

std::unordered_set<NodeTensor> getInitializer(const onnx::GraphProto &graph)
{
  std::unordered_set<NodeTensor> initializerNames;

  for (const auto &initializer : graph.initializer())
  {
    NodeTensor nt;
    nt.name = initializer.name();
    std::vector<int64_t> shape;

    for (const auto &dim : initializer.dims())
    {
      shape.push_back(dim);
    }

    nt.shape = shape;
    initializerNames.insert(nt);
  }

  return initializerNames;
}

std::unordered_set<NodeTensor> getIOvalue(const onnx::GraphProto &graph)
{
  std::unordered_set<NodeTensor> IOvalue;

  for (const auto &value_info : graph.value_info())
  {
    NodeTensor nt;
    nt.name = value_info.name();
    std::vector<int64_t> shape;

    for (const auto &dim : value_info.type().tensor_type().shape().dim())
    {
      shape.push_back(dim.dim_value());
    }

    nt.shape = shape;
    IOvalue.insert(nt);
  }

  for (auto value_info : graph.input())
  {
    NodeTensor nt;
    nt.name = value_info.name();
    std::vector<int64_t> shape;

    for (const auto &dim : value_info.type().tensor_type().shape().dim())
    {
      shape.push_back(dim.dim_value());
    }

    nt.shape = shape;
    IOvalue.insert(nt);
  }

  for (auto value_info : graph.output())
  {
    NodeTensor nt;
    nt.name = value_info.name();
    std::vector<int64_t> shape;

    for (const auto &dim : value_info.type().tensor_type().shape().dim())
    {
      shape.push_back(dim.dim_value());
    }

    nt.shape = shape;
    IOvalue.insert(nt);
  }

  return IOvalue;
}

std::unordered_set<NodeTensor>::const_iterator
isInputFromInitializer(const std::string &name, const std::unordered_set<NodeTensor> &tensors)
{
  return std::find_if(tensors.begin(), tensors.end(),
                      [&](const NodeTensor &tensor) { return tensor.name == name; });
}

void determineGraphInput(const onnx::GraphProto &g,
                         const std::unordered_set<NodeTensor> &initializerNames,
                         std::unordered_set<NodeTensor> &graphInputs)
{
  std::unordered_set<std::string> allnodeOutputs;

  // Iterate over each node in the graph to collect all outputs
  for (const auto &node : g.node())
  {
    // Get the output list of the current node
    const auto &outputs = node.output();

    // Insert each output into the set of all node outputs
    for (const auto &output : outputs)
    {
      allnodeOutputs.insert(output);
    }
  }

  // Iterate over each node in the graph to identify inputs not produced by any node
  for (const auto &node : g.node())
  {
    // Get the input list of the current node
    const auto &inputs = node.input();

    // Check each input to determine if it is an external input to the graph
    for (const auto &input : inputs)
    {
      // If the input is not found in the set of all node outputs, it is a graph input
      if (std::find(allnodeOutputs.begin(), allnodeOutputs.end(), input) == allnodeOutputs.end())
      {
        auto iter = isInputFromInitializer(input, initializerNames);
        NodeTensor nt;
        nt.name = input;

        if (iter != initializerNames.end())
        {
          graphInputs.insert(*iter);
        }
      }
    }
  }
}

void determineGraphOutput(const onnx::GraphProto &originalGraph, const onnx::GraphProto &g,
                          std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_1,
                          std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_2,
                          std::unordered_set<NodeTensor> &graphOutputs)
{
  auto allgraphInputs = allgraphInputs_1;
  allgraphInputs.insert(allgraphInputs.end(), allgraphInputs_2.begin(), allgraphInputs_2.end());

  for (const auto &node : g.node())
  {
    const auto &outputs = node.output();

    for (const auto &output : outputs)
    {
      int flag = 0;

      for (auto value_info : originalGraph.output())
      {
        if (value_info.name() == output)
        {
          NodeTensor nt;
          nt.name = value_info.name();
          std::cout << nt.name << std::endl;
          std::vector<int64_t> shape;

          for (const auto &dim : value_info.type().tensor_type().shape().dim())
          {
            shape.push_back(dim.dim_value());
          }

          nt.shape = shape;
          graphOutputs.insert(nt);
          flag = 1;

          break;
        }
      }

      if (flag)
      {
        continue;
      }

      for (size_t i = 0; i < allgraphInputs.size(); i++)
      {
        for (auto &input : allgraphInputs[i])
        {
          if (input.name == output)
          {
            graphOutputs.insert(input);
            flag = 1;

            break;
          }
        }

        if (flag)
        {
          break;
        }
      }
    }
  }
}

std::string findInputNode(const onnx::GraphProto &g, const std::string &outputTensorName)
{
  std::string node_name = "";

  for (const auto &node : g.node())
  {
    for (const auto &output : node.output())
    {
      if (output == outputTensorName)
      {
        node_name = node.name();
      }
    }
  }

  return node_name;
}

std::unordered_set<std::string> collectNodeNames(const onnx::GraphProto &graph)
{
  std::unordered_set<std::string> nodeNames;

  for (const auto &node : graph.node())
  {
    nodeNames.insert(node.name());
  }

  return nodeNames;
}

void mergeGraphs(onnx::GraphProto &targetGraph, onnx::GraphProto &sourceGraph)
{
  std::cout << "size before merged: " << targetGraph.node_size() << "+" << sourceGraph.node_size()
            << std::endl;
  int size_before = targetGraph.node_size() + sourceGraph.node_size();

  for (const auto &node : sourceGraph.node())
  {
    *targetGraph.add_node() = node;
  }

  std::cout << "size after merged: " << targetGraph.node_size() << std::endl;
  if (size_before != targetGraph.node_size())
  {
    std::cout << "error in mergeGraphs" << std::endl;
    std::exit(-1);
  }
}

onnx::GraphProto GetGraphFromOnnx(std::string &path)
{
  onnx::ModelProto model;

  std::ifstream input(path, std::ios::ate | std::ios::binary);
  if (!input.is_open())
  {
    std::cout << "Error: Failed to open file: " << path << std::endl;
    exit(-1);
  }

  // get current position in file
  std::streamsize size = input.tellg();

  // move to start of file
  input.seekg(0, std::ios::beg);

  // read raw data
  std::vector<char> buffer(size);
  input.read(buffer.data(), size);

  // parse protobuf
  model.ParseFromArray(buffer.data(), size);

  return model.graph();
}
