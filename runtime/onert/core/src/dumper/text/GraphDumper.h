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

namespace onert::ir
{
class Graph;
struct IOperation;
} // namespace onert::ir

namespace onert::compiler
{
class LoweredGraph;
} // namespace onert::compiler

namespace onert::compiler::train
{
class LoweredTrainableGraph;
} // namespace onert::compiler::train

namespace onert::dumper::text
{

std::string formatOperandBrief(ir::OperandIndex ind);
std::string formatOperand(const ir::Graph &, ir::OperandIndex ind);
std::string formatOperation(const ir::Graph &graph, ir::OperationIndex ind);
void dumpGraph(const ir::Graph &graph);
void dumpLoweredGraph(const compiler::LoweredGraph &lgraph);
void dumpLoweredGraph(const compiler::train::LoweredTrainableGraph &lgraph);

} // namespace onert::dumper::text

#endif // __ONERT_DUMPER_TEXT_GRAPH_DUMPER_H__
