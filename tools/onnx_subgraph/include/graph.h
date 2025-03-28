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
