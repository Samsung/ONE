/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_COMPILER_PASS_PERMUTATION_IO_PASS_H__
#define __ONERT_COMPILER_PASS_PERMUTATION_IO_PASS_H__

#include "Pass.h"
#include "compiler/ILoweredGraph.h"
#include "compiler/CompilerOptions.h"

namespace onert
{
namespace compiler
{
namespace pass
{

class PermutationIOPass : public Pass
{
public:
  PermutationIOPass(ILoweredGraph &lowered_graph, const CompilerOptions &options)
    : Pass{lowered_graph.graph()}, _lowered_graph(lowered_graph), _options(options)
  {
  }

public:
  std::string id() override { return "PermutationIOPass"; }
  void run() override;

private:
  void insertInputPermute(const ir::OperandIndex &index, const bool use_float,
                          const ir::Layout &from_layout);
  void insertOutputPermute(const ir::OperandIndex &index, const bool use_float,
                           const ir::Layout &to_layout);

private:
  compiler::ILoweredGraph &_lowered_graph;
  const compiler::CompilerOptions &_options;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_PERMUTATION_INSERTION_PASS_H__
