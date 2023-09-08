/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PROFILE_CIRCLE_NODE_ORIGIN_H__
#define __LUCI_PROFILE_CIRCLE_NODE_ORIGIN_H__

#include "CircleNodeID.h"

#include <luci/IR/CircleNode.h>

#include <set>

namespace luci
{

class CircleNodeOrigin
{
protected:
  struct Source
  {
  public:
    const std::string& name(void) const { return _name; }
    void name(const std::string &name) { _name = name; }

    uint32_t id(void) const { return _id; }
    void id(const uint32_t id) { _id = id; }

  private:
    std::string _name;
    uint32_t _id = 0;
  };

public:
  virtual std::set<const Source *> sources(void) const = 0;
};

std::shared_ptr<CircleNodeOrigin> single_origin(uint32_t id, const std::string &name);

std::shared_ptr<CircleNodeOrigin>
composite_origin(const std::initializer_list<std::shared_ptr<CircleNodeOrigin>> origins);

std::shared_ptr<CircleNodeOrigin>
composite_origin(const std::vector<std::shared_ptr<CircleNodeOrigin>> &origins);

} // namespace luci

namespace luci
{

bool has_origin(const luci::CircleNode *circle_node);

void add_origin(luci::CircleNode *circle_node, const std::shared_ptr<CircleNodeOrigin> origin);

// NOTE When circle_node does not have origin, nullptr is returned
const std::shared_ptr<luci::CircleNodeOrigin> get_origin(const luci::CircleNode *circle_node);

} // namespace luci

#endif // __LUCI_PROFILE_CIRCLE_NODE_ORIGIN_H__
