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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_REGISTRY_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_REGISTRY_H__

#include "GraphBuilderBase.h"

#include <map>

namespace luci
{

struct GraphBuilderSource
{
  virtual ~GraphBuilderSource() = default;

  /**
   * @brief Returns registered GraphBuilder pointer for operator (nullptr if not present)
   */
  virtual const GraphBuilderBase *lookup(const circle::BuiltinOperator &op) const = 0;
};

/**
 * @brief Class to return graph builder for Circle nodes
 */
class GraphBuilderRegistry final : public GraphBuilderSource
{
public:
  GraphBuilderRegistry();

public:
  GraphBuilderRegistry(const GraphBuilderSource *parent) : _parent{parent}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Returns registered GraphBuilder pointer for operator or
   *        nullptr if not registered
   */
  const GraphBuilderBase *lookup(const circle::BuiltinOperator &op) const final
  {
    if (_builder_map.find(op) == _builder_map.end())
      return (_parent == nullptr) ? nullptr : _parent->lookup(op);

    return _builder_map.at(op).get();
  }

  static GraphBuilderRegistry &get()
  {
    static GraphBuilderRegistry me;
    return me;
  }

public:
  void add(const circle::BuiltinOperator op, std::unique_ptr<GraphBuilderBase> &&builder)
  {
    _builder_map[op] = std::move(builder);
  }

private:
  const GraphBuilderSource *_parent = nullptr;

private:
  std::map<const circle::BuiltinOperator, std::unique_ptr<GraphBuilderBase>> _builder_map;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_REGISTRY_H__
