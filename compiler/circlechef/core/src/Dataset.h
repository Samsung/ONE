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

#ifndef __DATASET_H__
#define __DATASET_H__

#include <vector>

template <typename T> class Dataset
{
public:
  Dataset(const std::vector<T> &vec) : _vec{vec}
  {
    // DO NOTHING
  }

public:
  Dataset(std::vector<T> &&vec) : _vec{std::move(vec)}
  {
    // DO NOTHING
  }

public:
  template <typename Func> auto map(Func f) const -> Dataset<decltype(f(std::declval<T>()))>
  {
    using U = decltype(f(std::declval<T>()));
    std::vector<U> res;

    for (const auto &elem : _vec)
    {
      res.emplace_back(f(elem));
    }

    return Dataset<U>(std::move(res));
  }

public:
  const std::vector<T> &vectorize(void) const { return _vec; }

private:
  std::vector<T> _vec;
};

#endif // __DATASET_H__
