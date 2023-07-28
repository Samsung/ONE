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

#ifndef __ONERT_COMPILER_COMPILER_HELPERS_H__
#define __ONERT_COMPILER_COMPILER_HELPERS_H__

#include <compiler/ILoweredGraph.h>
#include <compiler/StaticShapeInferer.h>
#include <ir/Index.h>

#include <memory>
#include <unordered_map>

namespace onert
{
namespace compiler
{

/**
 * @brief     Create a shape inferer map for a lowered model
 * @param[in] lowered_subgs lowered model map
 * @return    Shape inferer map
 */
template <typename LoweredGraphType,
          typename = std::enable_if_t<std::is_base_of<ILoweredGraph, LoweredGraphType>::value>>
static std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>>
createStaticShapeInferers(
  const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<LoweredGraphType>> &lowered_subgs)
{
  std::unordered_map<ir::SubgraphIndex, ILoweredGraph *> lsubgs;
  for (auto &&e : lowered_subgs)
    lsubgs[e.first] = e.second.get();
  return StaticShapeInferer::createStaticShapeInferers(lsubgs);
}

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_COMPILER_HELPERS_H__
