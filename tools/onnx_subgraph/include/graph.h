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

#ifndef GRAPH_H
#define GRAPH_H

#include "onnx.pb.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
// save the size of each node's inputs and outputs
struct NodeIOSize
{
  std::vector<std::vector<int64_t>> inputSizes;
  std::vector<std::vector<int64_t>> outputSizes;
};

struct NodeTensor
{
  std::string name;
  std::vector<int64_t> shape;

  // Default constructor
  NodeTensor() = default;

  // Constructor with parameters
  NodeTensor(const std::string &n, const std::vector<int64_t> &s) : name(n), shape(s) {}

  // Equality comparison operator
  bool operator==(const NodeTensor &other) const
  {
    return name == other.name && shape == other.shape;
  }
};

namespace std
{
template <> struct hash<NodeTensor>
{
  size_t operator()(const NodeTensor &tensor) const
  {
    size_t hashValue = hash<string>()(tensor.name);
    for (auto &val : tensor.shape)
    {
      hashValue ^= hash<int64_t>()(val) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
    }
    return hashValue;
  }
};
} // namespace std
/**
 * @brief     Extracts the names and shapes of initializers from the ONNX graph.
 *
 * @param     [in] graph The ONNX graph from which to extract initializers.
 * @pre       The ONNX graph should be valid and contain initializers.
 * @post      The names and shapes of the initializers are stored in an unordered set of NodeTensor
 * objects.
 * @exception None
 * @return    An unordered set of NodeTensor objects containing the names and shapes of the
 * initializers.
 */
std::unordered_set<NodeTensor> getInitializer(const onnx::GraphProto &graph);
/**
 * @brief     Extracts the names and shapes of inputs, outputs, and value_info from the ONNX graph.
 *
 * @param     [in] graph The ONNX graph from which to extract inputs, outputs, and value_info.
 * @pre       The ONNX graph should be valid and contain inputs, outputs, and value_info.
 * @post      The names and shapes of the inputs, outputs, and value_info are stored in an unordered
 * set of NodeTensor objects.
 * @exception None
 * @return    An unordered set of NodeTensor objects containing the names and shapes of the inputs,
 * outputs, and value_info.
 */
std::unordered_set<NodeTensor> getIOvalue(const onnx::GraphProto &graph);
/**
 * @brief     Determines the input tensors of the graph that are not produced by any node in the
 * graph.
 *
 * @param     [in] g The ONNX GraphProto object representing the graph.
 * @param     [in] initializerNames A set of NodeTensor objects representing the initializers in the
 * graph.
 * @param     [out] graphInputs A set of NodeTensor objects representing the input tensors of the
 * graph.
 * @pre       The GraphProto object g should be valid and contain nodes with proper input and output
 * lists.
 * @post      The graphInputs set will be populated with NodeTensor objects that are inputs to the
 * graph.
 * @exception None
 * @return    None
 */
void determineGraphInput(const onnx::GraphProto &g,
                         const std::unordered_set<NodeTensor> &initializerNames,
                         std::unordered_set<NodeTensor> &graphInputs);
/**
 * @brief     Determines the output tensors of the graph that are either outputs of the original
 * graph or are used as inputs in other parts of the graph.
 *
 * @param     [in] originalGraph The original ONNX GraphProto object representing the graph.
 * @param     [in] g The ONNX GraphProto object representing the graph to analyze.
 * @param     [in] allgraphInputs_1 A vector of sets of NodeTensor objects representing the first
 * set of inputs to the graph.
 * @param     [in] allgraphInputs_2 A vector of sets of NodeTensor objects representing the second
 * set of inputs to the graph.
 * @param     [out] graphOutputs A set of NodeTensor objects representing the output tensors of the
 * graph.
 * @pre       The GraphProto objects originalGraph and g should be valid and contain nodes with
 * proper input and output lists.
 * @post      The graphOutputs set will be populated with NodeTensor objects that are outputs of the
 * graph.
 * @exception None
 * @return    None
 */
void determineGraphOutput(const onnx::GraphProto &originalGraph, const onnx::GraphProto &g,
                          std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_1,
                          std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_2,
                          std::unordered_set<NodeTensor> &graphOutputs);
/**
 * @brief     Finds the name of the node that produces a specified output tensor in the given ONNX
 * graph.
 *
 * @param     [in] g The ONNX GraphProto object representing the graph.
 * @param     [in] outputTensorName The name of the output tensor to find the producing node for.
 * @pre       The GraphProto object g should be valid and contain nodes with proper input and output
 * lists.
 * @post      None
 * @exception None
 * @return    The name of the node that produces the specified output tensor, or an empty string if
 * no such node is found.
 */
std::string findInputNode(const onnx::GraphProto &g, const std::string &outputTensorName);
/**
 * @brief     Collects the names of all nodes in the given ONNX graph.
 *
 * @param     [in] graph The ONNX GraphProto object representing the graph.
 * @pre       The GraphProto object graph should be valid and contain nodes with proper names.
 * @post      None
 * @exception None
 * @return    An unordered set containing the names of all nodes in the graph.
 */
std::unordered_set<std::string> collectNodeNames(const onnx::GraphProto &graph);
/**
 * @brief     Merges nodes from the source graph into the target graph.
 *
 * @param     [in,out] targetGraph The ONNX GraphProto object to which nodes will be added.
 * @param     [in] sourceGraph The ONNX GraphProto object from which nodes will be copied.
 * @pre       Both GraphProto objects should be valid.
 * @post      Nodes from sourceGraph are added to targetGraph.
 * @exception Exits the program with an error message if the number of nodes in targetGraph does not
 * match the expected size after merging.
 * @return    None
 */
void mergeGraphs(onnx::GraphProto &targetGraph, onnx::GraphProto &sourceGraph);

class Graph
{
private:
  /* data */
public:
  Graph() {}
  ~Graph() {}
  /**
   * @brief     Loads an ONNX model from a file and returns the graph contained within.
   *
   * @param     [in] path The file path to the ONNX model.
   * @pre       The file specified by path should exist and be a valid ONNX model.
   * @post      The ONNX model is parsed and its graph is returned.
   * @exception Exits the program with an error message if the file cannot be opened.
   * @return    The ONNX GraphProto object representing the graph from the model.
   */
  onnx::GraphProto GetGraphFromOnnx(std::string &path);
};
struct graph_adjacency_node
{
  std::vector<int> output_node_index;
  int rank;
  std::string name;
  int index;
};
#endif
