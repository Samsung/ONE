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

#include "record-hessian/HessianComputer.h"

#include <luci/IR/CircleQuantParam.h>

namespace record_hessian
{

void HessianComputer::update_qparam(
  const std::unordered_map<const luci::CircleNode *, HessianVector> *hessian_map)
{
  if (hessian_map == nullptr)
    throw std::invalid_argument("hessian_map is nullptr");

  for (auto iter = hessian_map->begin(); iter != hessian_map->end(); ++iter)
  {
    auto node = iter->first;
    auto hessian_vector = iter->second;

    auto quantparam = std::make_unique<luci::CircleQuantParam>();
    quantparam->hessian = hessian_vector.hessian;

    assert(node->quantparam() == nullptr);

    auto mutable_node = const_cast<luci::CircleNode *>(node);
    mutable_node->quantparam(std::move(quantparam));
  }
}

std::unique_ptr<HessianComputer> make_hessian_computer()
{
  return std::make_unique<HessianComputer>();
}

} // namespace record_hessian
