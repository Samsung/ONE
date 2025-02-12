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

#ifndef __ONERT_BACKEND_CL_COMMON_LIFETIME_MAP_H__
#define __ONERT_BACKEND_CL_COMMON_LIFETIME_MAP_H__

#include "cl_common/ParentInfo.h"

#include <ir/OperandIndexMap.h>

#include <map>
#include <vector>

namespace onert::backend::cl_common
{

// TODO Abstract UserType into LifetimeMap and LifetimeSeq
enum class UsesType
{
  FIRST,
  LAST
};

// TODO Define class or struct for LifetimeMap and LifetimeSeq
using LifetimeMap = std::map<size_t, std::pair<UsesType, ir::OperandIndex>>;
using LifetimeSeq = std::vector<std::pair<UsesType, ir::OperandIndex>>;

LifetimeMap createLifetimeMap(LifetimeSeq &seq, ir::OperandIndexMap<ParentInfo> &parent_map);

} // namespace onert::backend::cl_common

#endif // __ONERT_BACKEND_CL_COMMON_LIFETIME_MAP_H__
