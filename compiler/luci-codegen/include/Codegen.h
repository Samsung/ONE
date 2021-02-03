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

#ifndef NNCC_LUCI_CODEGEN_H
#define NNCC_LUCI_CODEGEN_H

#include "luci/IR/Module.h"
#include "luci/IR/CircleNodeDecl.h"

#include <memory>
#include <unordered_set>
#include <string>

namespace luci_codegen
{

struct Options
{
  /***
   * max size of constant buffer to inline in code in bytes
   */
  int max_inline_buffer_threshold = 1024;
};

class SubgraphContext;

class Codegen
{
public:
  Codegen(const Options &options = Options());

  ~Codegen();

  void process_module(luci::Module &module);

  void emit_code(std::string package_name);

private:

  bool fits_constrains(luci::CircleNode *node) const;

  std::vector<luci::CircleNode *> gather_suitable_nodes(luci::CircleNode *node);

  std::vector<std::vector<luci::CircleNode *>>
  extract_subgraphs(const std::vector<luci::CircleNode *> &nodes) const;

  SubgraphContext *create_subgraph(const std::vector<luci::CircleNode *> &nodes);

  void replace_subgraph_with_generated_node(SubgraphContext *subgraph) const;

  void cleanup_graph(SubgraphContext *subgraph) const;

  void process_graph(loco::Graph &graph);

  int _processed_graphs;
  Options _options;
  std::unordered_set<luci::CircleNode *> _processed;
  std::vector<SubgraphContext> _compiled_subgraphs;
};

} // namespace luci_codegen

#endif //NNCC_LUCI_CODEGEN_H
