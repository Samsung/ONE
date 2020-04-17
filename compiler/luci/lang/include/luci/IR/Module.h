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

#ifndef __LUCI_MODULE_H__
#define __LUCI_MODULE_H__

#include <loco/IR/Graph.h>

#include <memory>
#include <vector>

namespace luci
{

/**
 * @brief Collection of 'loco::Graph's
 */
class Module final
{
public:
  Module() = default;

  // Copy/Move is not allowed for Module
  Module(const Module &) = delete;
  Module(Module &&) = delete;

  ~Module() = default;

public:
  size_t size(void) const { return _graphs.size(); }

public:
  void add(std::unique_ptr<loco::Graph> &&g);

  /**
   * @brief provide main graph
   */
  loco::Graph *graph(void) const;

  /**
   * @brief provide graph with an index
   *
   * @note  graph(0) is interpreted as a main graph
   */
  loco::Graph *graph(size_t idx) const;

  // TODO provide graph accessor with a name

private:
  std::vector<std::unique_ptr<loco::Graph>> _graphs;
};

std::unique_ptr<Module> make_module(void);

} // namespace luci

#endif // __LUCI_MODULE_H__
