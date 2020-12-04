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

#ifndef __LUCI_REMOVE_REDUNDANT_RESHAPE_PASS_H__
#define __LUCI_REMOVE_REDUNDANT_RESHAPE_PASS_H__

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief  Class to Redundant Reshape node into 1 reshape node.
 */
struct RemoveRedundantReshapePass final : public logo::Pass
{
  const char *name(void) const final { return "luci::RemoveRedundantReshapePass"; }

  bool run(loco::Graph *g) final;
};

} // namespace luci

#endif // __LUCI_REMOVE_REDUNDANT_RESHAPE_PASS_H__
