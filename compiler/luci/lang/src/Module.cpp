/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Module.h"

#include <stdexcept>

namespace luci
{

void Module::remove(size_t idx)
{
  _graphs.at(idx).release();
  _graphs.erase(_graphs.begin() + idx);
}

void Module::add(std::unique_ptr<loco::Graph> &&g)
{
  if (g.get() == nullptr)
    throw std::invalid_argument("Module: Graph cannot be null");

  _graphs.emplace_back(std::move(g));
}

loco::Graph *Module::graph(void) const
{
  auto &graph = _graphs.at(0);
  return graph.get();
}

loco::Graph *Module::graph(size_t idx) const
{
  auto &graph = _graphs.at(idx);
  return graph.get();
}

std::unique_ptr<Module> make_module(void) { return std::make_unique<Module>(); }

} // namespace luci
