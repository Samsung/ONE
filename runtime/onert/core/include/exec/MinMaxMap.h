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
struct SMHash
{
  size_t operator()(const std::pair<ir::SubgraphIndex, ir::OperationIndex> &k) const noexcept
  {
    return std::hash<ir::SubgraphIndex>()(k.first) ^ std::hash<ir::OperationIndex>()(k.second);
  }
};
// SM means single model
using SMMinMaxMap = util::MinMaxMap<std::pair<ir::SubgraphIndex, ir::OperationIndex>, SMHash>;
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_MINMAX_MAP_H__
