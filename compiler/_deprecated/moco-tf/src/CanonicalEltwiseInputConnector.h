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

#ifndef __CANONICAL_ELTWISE_INPUT_CONNECTOR_H__
#define __CANONICAL_ELTWISE_INPUT_CONNECTOR_H__

#include <loco/IR/Node.h>

#include <utility>

namespace moco
{
namespace tf
{
namespace eltwise
{
namespace binary
{

using NodePair = std::pair<loco::Node *, loco::Node *>;

template <typename NodeTy> class InputConnector
{
public:
  InputConnector(NodeTy *node) : _node{node}
  {
    // DO NOTHING
  }

public:
  void operator()(const NodePair &p) const;

private:
  NodeTy *_node;
};

template <typename NodeTy> InputConnector<NodeTy> connect_to(NodeTy *node)
{
  return InputConnector<NodeTy>{node};
}

} // namespace binary
} // namespace eltwise
} // namespace tf
} // namespace moco

#endif // __CANONICAL_ELTWISE_INPUT_CONNECTOR_H__
