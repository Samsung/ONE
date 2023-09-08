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

#ifndef __ONERT_EXEC_MINMAX_MAP_H__
#define __ONERT_EXEC_MINMAX_MAP_H__

#include "ir/Index.h"
#include "util/MinMaxMap.h"

namespace onert
{
namespace exec
{

/*
 * NOTE: To record MinMax, the key would be better to use Tensor ID in circle.
 * But in general, onert does not keep track of circle tensor ID to onert tensor ID.
 * The ordering in tensors in onert may be different from ordering in circle
 * because onert could try optimization (reusing allocation, removing redundant tensors,
 * code optimization, ...)
 * For Linear Executor and CPU backcend, onert keep track of op index in generated Code.
 * MinMaxMap uses operation index instead.
 *
 * TODO: Stop recording in case of onert internal optimization (e.g. code fusion) occcurs.
 *       It rarely happens since most fusioning is done by compiler frontend, not by onert.
 */
struct OpMinMaxHash
{
  size_t operator()(const std::pair<ir::SubgraphIndex, ir::OperationIndex> &k) const noexcept
  {
    return std::hash<ir::SubgraphIndex>()(k.first) ^ std::hash<ir::OperationIndex>()(k.second);
  }
};
using OpMinMaxMap = util::MinMaxMap<std::pair<ir::SubgraphIndex, ir::OperationIndex>, OpMinMaxHash>;

struct IOMinMaxHash
{
  size_t operator()(const std::pair<ir::SubgraphIndex, ir::IOIndex> &k) const noexcept
  {
    return std::hash<ir::SubgraphIndex>()(k.first) ^ std::hash<ir::IOIndex>()(k.second);
  }
};
using IOMinMaxMap = util::MinMaxMap<std::pair<ir::SubgraphIndex, ir::IOIndex>, IOMinMaxHash>;
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_MINMAX_MAP_H__
