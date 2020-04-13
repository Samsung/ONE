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

#include "GraphBuilderContext.h"

namespace moco
{
namespace onnx
{

void SymbolTable::enroll(const std::string &node_name, loco::Node *node)
{
  MapNameNode_t::iterator iter = _namenode.find(node_name);

  if (iter != _namenode.end())
  {
    throw std::runtime_error{"Error: Duplicate node name in Graph: " + node_name};
  }

  _namenode[node_name] = node;
}

loco::Node *SymbolTable::node(const std::string &node_name)
{
  MapNameNode_t::iterator iter = _namenode.find(node_name);

  if (iter == _namenode.end())
  {
    throw std::runtime_error{"Error: Cannot find node with name in Graph: " + node_name};
  }

  return iter->second;
}

void SymbolTable::list(loco::Node *node, const std::string &name)
{
  MapNodeNames_t::iterator iter = _nodenames.find(node);

  if (iter == _nodenames.end())
  {
    // add a new vector for the first name
    _nodenames[node] = {name};
    return;
  }

  _nodenames[node].push_back(name);
}

unsigned SymbolTable::size(loco::Node *node)
{
  MapNodeNames_t::iterator iter = _nodenames.find(node);

  if (iter == _nodenames.end())
  {
    return 0;
  }

  return iter->second.size();
}

const std::string &SymbolTable::name(loco::Node *node, unsigned index)
{
  MapNodeNames_t::iterator iter = _nodenames.find(node);

  if (iter == _nodenames.end())
  {
    throw std::runtime_error{"Error: Cannot find names given node"};
  }

  if (index >= iter->second.size())
  {
    throw std::runtime_error{"Error: Invalid name index for given node"};
  }

  return iter->second.at(index);
}

} // namespace onnx
} // namespace moco
