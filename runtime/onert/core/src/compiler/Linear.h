/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_LINEAR_H__
#define __ONERT_COMPILER_LINEAR_H__

#include <vector>
#include <memory>

#include "ir/OpSequences.h"
#include "ir/Index.h"
#include "backend/ITensorBuilder.h"
#include "compiler/LoweredGraph.h"

namespace onert
{
namespace ir
{
struct OperationVisitor;
} // namespace ir
} // namespace onert

namespace onert
{
namespace compiler
{

class Linear
{
public:
  static std::vector<ir::OpSequenceIndex> linearize(const compiler::LoweredGraph &lowered_graph);
  static void dump(const compiler::LoweredGraph &lowered_graph,
                   const std::vector<ir::OpSequenceIndex> &order);
  static void planTensors(const compiler::LoweredGraph &lowered_graph,
                          const std::vector<ir::OpSequenceIndex> &order);

public:
  ir::OperandIndexMap<bool> _is_reshape;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_LINEAR_H__
