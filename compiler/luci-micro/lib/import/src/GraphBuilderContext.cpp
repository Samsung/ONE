/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/GraphBuilderContext.h"

#include <luci/Log.h>

#include <oops/UserExn.h>

namespace luci
{

void IndexNodeFinder::enroll(TensorIndex idx, CircleNode *node)
{
  if (_table.find(idx) != _table.end())
  {
    LOGGER(l);
    INFO(l) << "[luci] NodeFinder SKIP (" << idx << ") " << node << std::endl;
    return;
  }

  _table[idx] = node;
}

CircleNode *IndexNodeFinder::node(TensorIndex idx) const
{
  MapIndexNode_t::const_iterator iter = _table.find(idx);

  // dangle output node may exist that are not enrolled
  return (iter != _table.end()) ? iter->second : nullptr;
}

} // namespace luci
