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

#ifndef __LOCOEX_SERVICE_FORMATTED_GRAPH_H__
#define __LOCOEX_SERVICE_FORMATTED_GRAPH_H__

#include <locop/FormattedGraph.h>

#include <locoex/COpCall.h>

namespace locoex
{

class COpNodeSummaryBuilder final : public locop::NodeSummaryBuilder
{
public:
  COpNodeSummaryBuilder(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *node, locop::NodeSummary &s) const final;

private:
  bool summary(const locoex::COpCall *, locop::NodeSummary &) const;

private:
  const locop::SymbolTable *_tbl;
};

} // namespace locoex

#endif // __LOCOEX_SERVICE_FORMATTED_GRAPH_H__
