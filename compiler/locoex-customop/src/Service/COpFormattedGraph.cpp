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

#include "locoex/Service/COpFormattedGraph.h"

#include <locoex/COpCall.h>
#include <locoex/COpAttrTypes.h>
#include <locoex/COpDialect.h>

#include <pepper/str.h>

#include <sstream>
#include <stdexcept>

namespace locoex
{

bool COpNodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != locoex::COpDialect::get())
    return false;

  if (auto call_node = dynamic_cast<const locoex::COpCall *>(node))
  {
    return summary(call_node, s);
  }

  return false;
}

bool COpNodeSummaryBuilder::summary(const locoex::COpCall *node, locop::NodeSummary &s) const
{
  assert(node != nullptr);

  s.opname("COp.Call");
  for (uint32_t i = 0; i < node->arity(); i++)
    s.args().append(pepper::str("input_", i), _tbl->lookup(node->arg(i)));

  for (auto name : node->attr_names())
  {
    if (auto int_attr = node->attr<locoex::COpAttrType::Int>(name))
      s.args().append(name, pepper::str(int_attr->val()));
    else if (auto float_attr = node->attr<locoex::COpAttrType::Float>(name))
      s.args().append(name, pepper::str(float_attr->val()));
    else
      throw std::runtime_error("Not yet supported Attr Type");
  }

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

} // namespace locoex
