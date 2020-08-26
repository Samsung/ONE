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

#ifndef __SOUSCHEF_DIMS_H__
#define __SOUSCHEF_DIMS_H__

#include <functional>
#include <numeric>
#include <vector>

namespace souschef
{

template <typename T> using Dims = std::vector<T>;

template <typename SHAPETYPE> Dims<int32_t> as_dims(const SHAPETYPE &shape)
{
  std::vector<int32_t> res;

  for (auto &dim : shape.dim())
  {
    res.emplace_back(static_cast<int32_t>(dim));
  }

  return res;
}

int32_t element_count(const Dims<int32_t> &dims)
{
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int32_t>());
}

} // namespace souschef

#endif // __SOUSCHEF_DIMS_H__
