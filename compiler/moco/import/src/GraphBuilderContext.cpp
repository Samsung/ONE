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

#include "moco/Import/GraphBuilderContext.h"

#include <oops/UserExn.h>

#include <stdexcept>
#include <string>

namespace moco
{

void NodeDefTable::enroll(const std::string &node_name, const tensorflow::NodeDef *node)
{
  MapNameNode_t::iterator iter = _table.find(node_name);

  if (iter != _table.end())
  {
    throw oops::UserExn("Duplicate node name in GraphDef", node_name);
  }

  _table[node_name] = node;
}

const tensorflow::NodeDef *NodeDefTable::node(const std::string &node_name) const
{
  MapNameNode_t::const_iterator iter = _table.find(node_name);

  if (iter == _table.end())
  {
    throw oops::UserExn("Cannot find node with name in GraphDef", node_name);
  }

  return iter->second;
}

void SymbolTable::enroll(const TensorName &tensor_name, loco::Node *node)
{
  MapNameNode_t::iterator iter = _table.find(tensor_name);

  if (iter != _table.end())
  {
    throw oops::UserExn("Duplicate node name in GraphDef", tensor_name.name());
  }

  _table[tensor_name] = node;
}

loco::Node *SymbolTable::node(const TensorName &tensor_name) const
{
  MapNameNode_t::const_iterator iter = _table.find(tensor_name);

  if (iter == _table.end())
  {
    throw oops::UserExn("Cannot find node with name in GraphDef", tensor_name.name());
  }

  return iter->second;
}

void UpdateQueue::enroll(std::unique_ptr<GraphUpdate> &&update)
{
  _queue.push_back(std::move(update));
}

} // namespace moco
