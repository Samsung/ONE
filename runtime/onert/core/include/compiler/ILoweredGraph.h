/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_ILOWERED_GRAPH_H__
#define __ONERT_COMPILER_ILOWERED_GRAPH_H__

#include "ir/Graph.h"
#include "compiler/GraphLowerInfo.h"

namespace onert::compiler
{

struct ILoweredGraph
{
  virtual ~ILoweredGraph() = default;
  virtual ir::Graph &graph() = 0;
  virtual const ir::Graph &graph() const = 0;
  virtual const compiler::GraphLowerInfo &lower_info() const = 0;
  virtual compiler::GraphLowerInfo &lower_info() = 0;
  virtual void setHasDynamicTensor(ir::OperationIndex ind, bool val) = 0;
  virtual bool getHasDynamicTensor(ir::OperationIndex ind) const = 0;
};

} // namespace onert::compiler

#endif // __ONERT_COMPILER_ILOWERED_GRAPH_H__
