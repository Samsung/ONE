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

#ifndef __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_REGISTRY_H__
#define __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_REGISTRY_H__

#include "GraphBuilder.h"

#include <map>

namespace moco
{
namespace onnx
{

/**
 * @brief Class to return graph builder for passed onnx Operator
 */
class GraphBuilderRegistry
{
public:
  /**
   * @brief Returns registered GraphBuilder pointer for operator or
   *        nullptr if not registered
   */
  const GraphBuilder *lookup(const std::string &op) const
  {
    if (_builder_map.find(op) == _builder_map.end())
      return nullptr;

    return _builder_map.at(op).get();
  }

  static GraphBuilderRegistry &get()
  {
    static GraphBuilderRegistry me;
    return me;
  }

public:
  void add(const std::string op, std::unique_ptr<GraphBuilder> &&builder)
  {
    _builder_map[op] = std::move(builder);
  }

private:
  std::map<const std::string, std::unique_ptr<GraphBuilder>> _builder_map;
};

} // namespace onnx
} // namespace moco

#include <memory>

#define REGISTER_OP_BUILDER(NAME, BUILDER)                                                  \
  namespace                                                                                 \
  {                                                                                         \
  __attribute__((constructor)) void reg_op(void)                                            \
  {                                                                                         \
    std::unique_ptr<moco::onnx::BUILDER> builder = std::make_unique<moco::onnx::BUILDER>(); \
    moco::onnx::GraphBuilderRegistry::get().add(#NAME, std::move(builder));                 \
  }                                                                                         \
  }

#endif // __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_REGISTRY_H__
