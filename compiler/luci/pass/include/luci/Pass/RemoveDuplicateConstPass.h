/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_REMOVE_DUPLICATE_CONST_PASS_H__
#define __LUCI_REMOVE_DUPLICATE_CONST_PASS_H__

#include <luci/IR/CircleNodes.h>
#include <logo/Pass.h>

namespace luci
{

/**
 * @brief  Class to remove duplicate Const nodes.
 */
struct RemoveDuplicateConstPass final : public logo::Pass
{
  const char *name(void) const final { return "luci::RemoveDuplicateConstPass"; }

  bool run(loco::Graph *g) final;

private:
  bool remove_duplicate_const();

  template <loco::DataType DT> void add_to_map(luci::CircleConst *const_node);

  std::map<float, std::vector<CircleConst *>> _sum_to_const;
};

} // namespace luci

#endif // __LUCI_REMOVE_DUPLICATE_CONST_PASS_H__
