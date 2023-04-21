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

#ifndef __CIRCLE_OPSELECTOR_OPSELECTOR_H__
#define __CIRCLE_OPSELECTOR_OPSELECTOR_H__

#include "SelectType.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodeDecl.h>

#include <string>
#include <vector>

namespace opselector
{

class OpSelector final
{
private:
  const luci::Module *_module;

public:
  OpSelector(const luci::Module *module);

private:
  template <SelectType SELECT_TYPE>
  std::vector<const luci::CircleNode *> select_nodes_by(const std::vector<std::string> &tokens,
                                                        const uint32_t graph_idx);

public:
  template <SelectType SELECT_TYPE>
  std::unique_ptr<luci::Module> select_by(const std::vector<std::string> &inputs);
};

extern template std::unique_ptr<luci::Module>
OpSelector::select_by<SelectType::ID>(const std::vector<std::string> &inputs);
extern template std::unique_ptr<luci::Module>
OpSelector::select_by<SelectType::NAME>(const std::vector<std::string> &inputs);

} // namespace opselector

#endif // __CIRCLE_OPSELECTOR_OPSELECTOR_H__
