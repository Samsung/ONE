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

#ifndef __RECORD_HESSIAN_HESSIANVECTOR_H__
#define __RECORD_HESSIAN_HESSIANVECTOR_H__

#include <vector>
#include <cstddef>
#include <unordered_map>
#include <luci/IR/CircleNodes.h>

namespace record_hessian
{
using HessianMap = std::unordered_map<const luci::CircleNode *, std::vector<float>>;

struct HessianVector
{
  std::vector<float> hessian;
  size_t count;

  HessianVector() : count(0) {}

  void update(const std::vector<float> &new_hessian)
  {
    if (count == 0)
    {
      hessian.resize(new_hessian.size());
    }
    else if (hessian.size() != new_hessian.size())
    {
      hessian.resize(new_hessian.size());
    }

    size_t numel = new_hessian.size();
    float alpha = 1.f / static_cast<float>(count + 1);

    for (size_t i = 0; i < numel; ++i)
    {
      hessian[i] = (hessian[i] * count + new_hessian[i]) * alpha;
    }

    count++;
  };
};

} // namespace record_hessian

#endif // __RECORD_HESSIAN_HESSIANVECTOR_H__
