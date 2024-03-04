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

#include <map>
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

public:
  void source_table(const std::map<uint32_t, std::string> &table) { _source_table = table; }

  const std::map<uint32_t, std::string> &source_table(void) const { return _source_table; }

  void map_tenros_indexes(const std::map<uint32_t, uint32_t> &map_tenros_indexes) { _map_tensors_indexes = map_tenros_indexes; }

  const std::map<uint32_t, uint32_t> &map_tenros_indexes(void) const { return _map_tensors_indexes; }

private:
  std::vector<std::unique_ptr<loco::Graph>> _graphs;

private:
  /**
   * @brief Metadata about source table for profiling
   *
   * @note  Key is ID of node and value is name of node.
   *
   *        If there was originally imported 'source_table' in circle model,
   *        the table will be stored as it is.
   *        Otherwise, new 'source_table' is created with imported nodes.
   *
   *        Even if Module has multiple subgraphs, only first subgraph is considered.
   */
  std::map<uint32_t, std::string> _source_table;

  std::map<uint32_t, uint32_t> _map_tensors_indexes;
};

std::unique_ptr<Module> make_module(void);

} // namespace luci

#endif // __LUCI_MODULE_H__
