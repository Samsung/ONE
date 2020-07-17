/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file Node.h
 * @brief    This file contains Node class
 * @ingroup  COM_AI_RUNTIME
 *
 */

#ifndef __ONERT_DUMPER_DOT_NODE_H__
#define __ONERT_DUMPER_DOT_NODE_H__

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace onert
{
namespace dumper
{
namespace dot
{

enum BGCOLORS : int
{
  RED,
  BLUE,
  GREEN,
  PUPLE,
  ORANGE,
  YELLOW,
  BROWN,
  PINK
};

/**
 * @brief Class that represents a Node in "dot" format
 *
 */
class Node
{
public:
  const static std::string DEFAULT_FILLCOLOR;
  const static std::string DEFAULT_COLORSCHEME;
  const static std::string BG_COLORS[8];

public:
  /**
   * @brief Destroy the Node object
   *
   */
  virtual ~Node() = default;

  /**
   * @brief Construct a new Node object
   *
   * @param id
   */
  Node(const std::string &id);

  /**
   * @brief return id
   *
   * @return id
   */
  std::string id() const { return _id; }

  /**
   * @brief return attributes
   *
   * @return const reference of attributes object
   */
  const std::unordered_map<std::string, std::string> &attributes() const { return _attributes; }
  /**
   * @brief Store an attribute with key-value pair
   *
   * @param[in] key attribute's key
   * @param[in] val attribute's value that is associated with the key
   */
  void setAttribute(const std::string &key, const std::string &val);
  /**
   * @brief Get the attributte value that is associated with key
   *
   * @param[in] key key of the attribute
   * @return value that is associated with the key
   */
  std::string getAttribute(const std::string &key);

  /**
   * @brief Add an edge in the graph, which is an outgoing edge
   *
   * @param[in] dotinfo A node that the new edge will be connected to
   */
  void addOutEdge(Node *dotinfo) { _out_edges.emplace_back(dotinfo); }
  /**
   * @brief Return list of out edges
   *
   * @return Edges
   */
  const std::vector<Node *> &out_edges() const { return _out_edges; }

private:
  std::string _id;
  std::unordered_map<std::string, std::string> _attributes;
  std::vector<Node *> _out_edges;
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_NODE_H__
