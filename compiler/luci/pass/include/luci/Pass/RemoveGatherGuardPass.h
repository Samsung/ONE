/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_REMOVE_GATHER_GUARD_PASS_H__
#define __LUCI_REMOVE_GATHER_GUARD_PASS_H__

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief Class to remove Add+FloorMod guard ops of Gather
 * @note  If the indices of Gather is guarenteed to be positive by the user,
 *        Add/FloorMod guard ops can be removed.
 *        This pass is to remove Add+FloorMod having INT32/INT64 dtypes
 *        for some backends cannot process this in quantized models.
 */
struct RemoveGatherGuardPass final : public logo::Pass
{
  const char *name(void) const final { return "luci::RemoveGatherGuardPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace luci

#endif // __LUCI_REMOVE_GATHER_GUARD_PASS_H__
