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

class CodegenContext;

class LuciCodegen
{
public:
  LuciCodegen(const Options &options = Options());

  ~LuciCodegen();

  void add_operator(luci::CircleNode *node);

  bool supported(luci::CircleNode *node);

  void process(loco::Graph &graph);

  void process(luci::Module &module);

  void emit_code(std::string package_name);
private:
  std::unique_ptr<CodegenContext> _context;
  Options _options;
};

} // namespace luci_codegen

#endif //NNCC_LUCI_CODEGEN_H
