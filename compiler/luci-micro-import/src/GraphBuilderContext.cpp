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
  auto iter = _table.find(idx);
  if (iter != _table.end())
  {
    LOGGER(l);
    INFO(l) << "[luci] NodeFinder SKIP (" << idx << ") " << node << ":" << node->name()
            << " existing: " << iter->second << ":" << iter->second->name() << std::endl;
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

void IndexTensorOutputs::enroll(TensorIndex idx)
{
  auto iter = _set.find(idx);
  if (iter != _set.end())
  {
    LOGGER(l);
    INFO(l) << "[luci] TensorOutputs SKIP (" << idx << ") existing" << std::endl;
    return;
  }
  _set.insert(idx);
}

bool IndexTensorOutputs::find(TensorIndex idx) { return (_set.find(idx) != _set.end()); }

} // namespace luci
