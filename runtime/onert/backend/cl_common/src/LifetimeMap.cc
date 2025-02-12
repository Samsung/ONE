/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "cl_common/LifetimeMap.h"

#include <unordered_map>

namespace onert::backend::cl_common
{

LifetimeMap createLifetimeMap(LifetimeSeq &lifetime_seq,
                              ir::OperandIndexMap<ParentInfo> &parent_map)
{
  // Update lifetime sequence to apply subtensor optimization
  std::unordered_map<ir::OperandIndex, ir::OperandIndex> root_map;
  std::function<ir::OperandIndex &(ir::OperandIndex)> find_root =
    [&](ir::OperandIndex ind) -> ir::OperandIndex & {
    ir::OperandIndex &ret = root_map[ind];

    // We know the root parent value already
    if (ret.valid())
      return ret;

    auto itr = parent_map.find(ind);
    if (itr == parent_map.end())
    {
      // If there is no parent, let's store the value of itself
      return ret = ind;
    }
    else
    {
      return ret = find_root(itr->second.parent);
    }
  };

  ir::OperandIndexMap<bool> first_use_check;
  ir::OperandIndexMap<bool> last_use_check;
  LifetimeMap lifetime_map;
  for (size_t i = 0; i < lifetime_seq.size(); i++)
  {
    const auto &[entry_uses_type, entry_idx] = lifetime_seq[i];
    if (entry_uses_type != UsesType::FIRST)
      continue;
    auto root_ind = find_root(entry_idx);
    if (first_use_check[root_ind])
      continue;
    first_use_check[root_ind] = true;
    lifetime_map[i] = {UsesType::FIRST, root_ind};
  }

  for (int i = lifetime_seq.size() - 1; i >= 0; i--)
  {
    const auto &[entry_uses_type, entry_idx] = lifetime_seq[i];
    if (entry_uses_type != UsesType::LAST)
      continue;
    auto root_ind = find_root(entry_idx);
    if (last_use_check[root_ind])
      continue;
    last_use_check[root_ind] = true;
    lifetime_map[i] = {UsesType::LAST, root_ind};
  }

  return lifetime_map;
}

} // namespace onert::backend::cl_common
