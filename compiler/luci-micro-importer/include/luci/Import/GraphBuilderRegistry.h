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

#include <vector>

namespace luci
{

struct GraphBuilderSource
{
  virtual ~GraphBuilderSource() = default;

  /**
   * @brief Returns registered GraphBuilder pointer for operator (nullptr if not present)
   */
  virtual const GraphBuilderBase *lookup(circle::BuiltinOperator op) const = 0;
};

/**
 * @brief Class to return graph builder for Circle nodes
 */
class GraphBuilderRegistry final : public GraphBuilderSource
{
public:
  GraphBuilderRegistry();

public:
  /**
   * @brief Returns registered GraphBuilder pointer for operator or
   *        nullptr if not registered
   */
  const GraphBuilderBase *lookup(circle::BuiltinOperator op) const final
  {
    return _op2builder.at(uint32_t(op)).get();
  }

  static GraphBuilderRegistry &get()
  {
    static GraphBuilderRegistry me;
    return me;
  }

public:
  void add(const circle::BuiltinOperator op, std::unique_ptr<GraphBuilderBase> &&builder)
  {
    _op2builder[uint32_t(op)] = std::move(builder);
  }

private:
  std::vector<std::unique_ptr<GraphBuilderBase>> _op2builder;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_REGISTRY_H__
