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

#include "luci/Profile/CircleNodeOrigin.h"

#include <loco.h>

#include <cassert>
#include <vector>

namespace
{

/**
 * @brief Set annotation for origin information
 * @note  Once CircleNodeOrigin is annotated, it should not be changed.
 *        If CircleNodeOrigin is needed to be changed, create new CircleNodeOrigin.
 */
class CircleNodeOriginAnnotation final : public loco::NodeAnnotation
{
public:
  CircleNodeOriginAnnotation() = delete;

  CircleNodeOriginAnnotation(const std::shared_ptr<luci::CircleNodeOrigin> origin) : _origin(origin)
  {
    // Do nothing
  }

public:
  const std::shared_ptr<luci::CircleNodeOrigin> origin(void) const { return _origin; }
  // No setter

private:
  const std::shared_ptr<luci::CircleNodeOrigin> _origin;
};

} // namespace

namespace
{

class SingleOrigin final : public luci::CircleNodeOrigin
{
public:
  SingleOrigin() = delete;

  SingleOrigin(uint32_t id, std::string name)
  {
    _source.id(id);
    _source.name(name);
  }

public:
  std::set<const Source *> sources(void) const final
  {
    std::set<const Source *> res;
    res.emplace(&_source);
    return res;
  }

private:
  Source _source;
};

class CompositeOrigin final : public luci::CircleNodeOrigin
{
public:
  CompositeOrigin() = delete;

  template <typename T> CompositeOrigin(T origins)
  {
    if (origins.size() == 0)
      throw std::invalid_argument("No origins provided");

    for (auto &origin : origins)
    {
      if (origin != nullptr)
        _origins.emplace_back(origin);
    }
  }

public:
  std::set<const Source *> sources(void) const final
  {
    std::set<const Source *> res;

    for (auto &origin : _origins)
    {
      for (auto source : origin->sources())
      {
        res.emplace(source);
      }
    }

    return res;
  }

private:
  std::vector<std::shared_ptr<CircleNodeOrigin>> _origins;
};

} // namespace

namespace luci
{

std::shared_ptr<CircleNodeOrigin> single_origin(uint32_t id, std::string name)
{
  return std::make_shared<SingleOrigin>(id, name);
}

std::shared_ptr<CircleNodeOrigin>
composite_origin(std::initializer_list<std::shared_ptr<CircleNodeOrigin>> origins)
{
  return std::make_shared<CompositeOrigin>(origins);
}

std::shared_ptr<CircleNodeOrigin>
composite_origin(std::vector<std::shared_ptr<CircleNodeOrigin>> origins)
{
  return std::make_shared<CompositeOrigin>(origins);
}

} // namespace luci

namespace luci
{

bool has_origin(const luci::CircleNode *circle_node)
{
  return circle_node->annot<CircleNodeOriginAnnotation>() != nullptr;
}

void add_origin(luci::CircleNode *circle_node, const std::shared_ptr<CircleNodeOrigin> origin)
{
  circle_node->annot<CircleNodeOriginAnnotation>(nullptr);
  circle_node->annot(std::make_unique<CircleNodeOriginAnnotation>(origin));
}

const std::shared_ptr<luci::CircleNodeOrigin> get_origin(const luci::CircleNode *circle_node)
{
  if (!has_origin(circle_node))
    return nullptr;

  return circle_node->annot<CircleNodeOriginAnnotation>()->origin();
}

} // namespace luci
