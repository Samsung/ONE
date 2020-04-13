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

#ifndef __LOCOP_FORMATTED_GRAPH_H__
#define __LOCOP_FORMATTED_GRAPH_H__

#include "locop/SymbolTable.h"
#include "locop/NodeSummary.h"
#include "locop/NodeSummaryBuilder.h"
// TODO Remove this redundant include
#include "locop/CanonicalNodeSummaryBuilder.h"

#include <loco.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace locop
{

struct FormattedGraph
{
  virtual ~FormattedGraph() = default;

  virtual void dump(std::ostream &os) const = 0;
};

std::ostream &operator<<(std::ostream &, const FormattedGraph &);

enum Formatter
{
  LinearV1,
  // TO BE ADDED
};

template <Formatter F> class FormattedGraphImpl;

template <> class FormattedGraphImpl<Formatter::LinearV1> final : public FormattedGraph
{
public:
  FormattedGraphImpl(loco::Graph *graph) : _graph{graph} {}

public:
  void dump(std::ostream &os) const final;

public:
  FormattedGraphImpl<Formatter::LinearV1> &with(std::unique_ptr<NodeSummaryBuilderFactory> &&f)
  {
    _factory = std::move(f);
    return (*this);
  }

private:
  loco::Graph *_graph;

  /**
   * @brief User-provided NodeSummaryBuilderFactory
   */
  std::unique_ptr<NodeSummaryBuilderFactory> _factory = nullptr;
};

template <Formatter F> FormattedGraphImpl<F> fmt(loco::Graph *g)
{
  return FormattedGraphImpl<F>{g};
}

template <Formatter F> FormattedGraphImpl<F> fmt(const std::unique_ptr<loco::Graph> &g)
{
  return fmt<F>(g.get());
}

} // namespace locop

#endif // __LOCOP_FORMATTED_GRAPH_H__
