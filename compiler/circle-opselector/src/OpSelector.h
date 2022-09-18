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

#ifndef __CIRCLE_OPSELECTOR_OPSELECTOR_H__
#define __CIRCLE_OPSELECTOR_OPSELECTOR_H__

#include <luci/IR/Module.h>

#include <luci/Importer.h>

#include <vector>

namespace luci
{

void post_import_graph(luci::Module *module, const luci::CircleReader &reader);

}

namespace opselector
{

/**
 * @brief  OpSelector creates a new graph consisting of the selected nodes.
 *
 * @note   It find graph's input and output node, and build graph.
 */
class OpSelector
{
public:
  OpSelector() = default;
  OpSelector(std::vector<char> &model_data) { _reader.parse(circle::GetModel(model_data.data())); }
  ~OpSelector() = default;

public:
  void find_unconnected_nodes(std::vector<const luci::CircleNode *> &selected_nodes,
                              std::set<uint32_t> &used_output_tensors,
                              std::set<uint32_t> &graph_inputs, std::set<uint32_t> &graph_outputs);
  void print_selected_nodes(std::vector<const luci::CircleNode *> selected_nodes);
  std::unique_ptr<luci::Module> select_nodes(std::vector<const luci::CircleNode *> selected_nodes);

private:
  void build_cache_outputs(luci::GraphBuilderContext &gb_context);
  void create_graph_inputs(luci::GraphBuilderContext &gb_context, uint32_t input);
  void create_circle_const(luci::GraphBuilderContext &gb_context);
  void import_operators(luci::GraphBuilderContext &gb_context);
  void create_graph_outputs(luci::GraphBuilderContext &gb_context, uint32_t output);

private:
  luci::CircleReader _reader;
  bool _has_subgraph = false; // A flag indicating whether to copy the subgraph or not,
};

} // namespace opselector

#endif // __CIRCLE_OPSELECTOR_OPSELECTOR_H__
