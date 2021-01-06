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

#ifndef __ONERT_DUMPER_DOT_DOT_BUILDER_H__
#define __ONERT_DUMPER_DOT_DOT_BUILDER_H__

#include <sstream>

#include "ir/Index.h"
#include "ir/Operation.h"
#include "ir/Operand.h"

#include "OperationNode.h"
#include "OperandNode.h"

using Operation = onert::ir::Operation;
using Object = onert::ir::Operand;

namespace onert
{
namespace dumper
{
namespace dot
{

class DotBuilder
{
public:
  DotBuilder();

public:
  void update(const Node &dotinfo);

  void writeDot(std::ostream &os);

private:
  void add(const Node &dotinfo);
  void addEdge(const Node &dotinfo1, const Node &dotinfo2);

  std::stringstream _dot;
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_DOT_BUILDER_H__
