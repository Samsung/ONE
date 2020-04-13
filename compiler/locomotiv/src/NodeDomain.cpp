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

#include "NodeDomain.h"

#include <cassert>

namespace locomotiv
{

struct NodeDomain final : public loco::NodeAnnotation
{
  NodeDomain(const loco::Domain &domain) : value(domain)
  {
    // DO NOTHING
  }

  loco::Domain value = loco::Domain::Unknown;
};

void annot_domain(loco::Node *node, const loco::Domain &domain)
{
  assert(domain != loco::Domain::Unknown);
  auto node_domain = std::unique_ptr<NodeDomain>(new NodeDomain(domain));
  assert(node_domain);
  node->annot(std::move(node_domain));
}

loco::Domain annot_domain(const loco::Node *node)
{
  auto node_domain = node->annot<NodeDomain>();
  if (node_domain)
    return node_domain->value;
  else
    return loco::Domain::Unknown;
}

void erase_annot_domain(loco::Node *node) { node->annot<NodeDomain>(nullptr); }

} // namespace locomotiv
