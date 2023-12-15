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

#include "LayoutResolver.h"

#include "TransposeEliminationPass.h"
#include "TransposeInsertionPass.h"

#include <util/ConfigSource.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

void LayoutResolver::operator()(backend::ContextData &data)
{
  auto &graph = *data.graph;
  const auto backend_layout = backendLayout();
  if (checkAllOfLegalLayout(graph, backend_layout))
    return;

  insertTransposeOps(data);
  removeTwofoldTransposeOps(data);

  assert(checkAllOfLegalLayout(graph, backend_layout));
}

void LayoutResolver::insertTransposeOps(backend::ContextData &data)
{
  auto &graph = *data.graph;

  TransposeInsertionPass pass{graph};
  pass.run(backendLayout());

  data.op_order = graph.topolSortOperations();
}

void LayoutResolver::removeTwofoldTransposeOps(backend::ContextData &data)
{
  auto &graph = *data.graph;

  TransposeEliminationPass pass{graph};
  pass.run();

  data.op_order = graph.topolSortOperations();
}

ir::Layout LayoutResolver::backendLayout() const
{
  const std::string acl_layout_str = util::getConfigString(util::config::ACL_LAYOUT);
  if (acl_layout_str == "NHWC")
  {
    return ir::Layout::NHWC;
  }
  else if (acl_layout_str == "NCHW")
  {
    return ir::Layout::NCHW;
  }

  return ir::Layout::UNKNOWN;
}

bool LayoutResolver::checkAllOfLegalLayout(const ir::Graph &graph, ir::Layout backend_layout) const
{
  bool is_legal = true;
  graph.operands().iterate([&](const onert::ir::OperandIndex &, const onert::ir::Operand &operand) {
    const auto layout = operand.info().layout();
    if (layout != ir::Layout::UNKNOWN && layout != backend_layout)
    {
      // TODO Check if operand is a input/output transpose op
      is_legal = false;
    }
  });

  return is_legal;
}

} // namespace acl_common
} // namespace backend
} // namespace onert
