/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_DUMPER_TEXT_GRAPH_DUMPER_H__
#define __ONERT_DUMPER_TEXT_GRAPH_DUMPER_H__

#include <ir/Index.h>

namespace onert
{
namespace ir
{
class Graph;
struct IOperation;
} // namespace ir
} // namespace onert

namespace onert
{
namespace compiler
{
class LoweredGraph;

namespace train
{
class LoweredTrainableGraph;
} // namespace train
} // namespace compiler
} // namespace onert

namespace onert
{
namespace dumper
{
namespace text
{

std::string formatOperandBrief(ir::OperandIndex ind);
std::string formatOperand(const ir::Graph &, ir::OperandIndex ind);
std::string formatOperation(const ir::Graph &graph, ir::OperationIndex ind);
void dumpGraph(const ir::Graph &graph);
void dumpLoweredGraph(const compiler::LoweredGraph &lgraph);
void dumpLoweredGraph(const compiler::train::LoweredTrainableGraph &lgraph);

} // namespace text
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_TEXT_GRAPH_DUMPER_H__
