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

#ifndef NNCC_KERNELBUILDER_H
#define NNCC_KERNELBUILDER_H

#include "luci/IR/CircleNodeDecl.h"
#include "SubgraphContext.h"

#include "luci/IR/CircleNodeVisitor.h"

namespace luci_codegen
{

bool is_supported(luci::CircleNode *node);

class CodegenKernelBuilder: public luci::CircleNodeMutableVisitor<void>
{
private:
  SubgraphContext &_subgraph;

  // elementwise operator supports

  template <typename OP>
  void binary_operator(luci::CircleNode *node);

public:
  Halide::Func get_func(luci::CircleNode *node);

  explicit CodegenKernelBuilder(SubgraphContext &subgraph);

  void visit(luci::CircleConst *node) override;

  void visit(luci::CircleAdd *node) override;

  void visit(luci::CircleSub *node) override;

  void visit(luci::CircleMul *node) override;

  void visit(luci::CircleDiv *node) override;

  /// @brief Default fallback
  void visit(luci::CircleNode *) override;
};

}

#endif // NNCC_KERNELBUILDER_H
