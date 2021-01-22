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

#ifndef NNCC_SUBGRAPHCONTEXT_H
#define NNCC_SUBGRAPHCONTEXT_H

#include "luci/IR/CircleNodeDecl.h"

#include "Halide.h"

namespace luci_codegen
{

class SubgraphContext
{
public:
  SubgraphContext() {}

  std::map<luci::CircleNode *, Halide::Func> &generated_funcs() { return _generated_funcs; }

  std::vector<Halide::Argument> inputs() { return _inputs; }

  std::vector<Halide::Func> outputs()
  {
    //NYI
  }

  void add_input(Halide::Argument input) { _inputs.push_back(input); }

private:
  std::map<luci::CircleNode *, Halide::Func> _generated_funcs;
  std::vector<Halide::Argument> _inputs;
};

}

#endif //NNCC_SUBGRAPHCONTEXT_H
