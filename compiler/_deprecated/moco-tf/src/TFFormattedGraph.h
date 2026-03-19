/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TF_FORMATTED_GRAPH_H__
#define __TF_FORMATTED_GRAPH_H__

#include <locop/FormattedGraph.h>

#include <memory>

namespace moco
{
namespace tf
{

class MocoNodeSummaryBuilder final : public locop::NodeSummaryBuilder
{
public:
  MocoNodeSummaryBuilder(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *node, locop::NodeSummary &s) const final;

private:
  const locop::SymbolTable *_tbl;
};

class TFNodeSummaryBuilderFactory final : public locop::NodeSummaryBuilderFactory
{
public:
  TFNodeSummaryBuilderFactory() = default;

public:
  std::unique_ptr<locop::NodeSummaryBuilder> create(const locop::SymbolTable *tlb) const final
  {
    return std::make_unique<MocoNodeSummaryBuilder>(tlb);
  }
};

} // namespace tf
} // namespace moco

#endif // __TF_FORMATTED_GRAPH_H__
