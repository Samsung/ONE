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

#include "ir/Index.h"
#include "compiler/ILoweredGraph.h"

namespace onert
{
namespace compiler
{

class Linear
{
public:
  static std::vector<ir::OperationIndex> linearize(const compiler::ILoweredGraph &lowered_graph);
  static std::vector<ir::OperationIndex> blinearize(const compiler::ILoweredGraph &lowered_graph);
  static void dump(const compiler::ILoweredGraph &lowered_graph,
                   const std::vector<ir::OperationIndex> &order);
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_LINEAR_H__
