/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __BINO_H__
#define __BINO_H__

#include <utility>

namespace bino
{

template <typename Callable> class UniformTransform
{
public:
  UniformTransform(Callable &&cb) : f{std::forward<Callable>(cb)}
  {
    // DO NOTHING
  }

public:
  template <typename T>
  auto operator()(const std::pair<T, T> &p) const
    -> decltype(std::make_pair(std::declval<Callable>()(p.first),
                               std::declval<Callable>()(p.second)))
  {
    return std::make_pair(f(p.first), f(p.second));
  }

private:
  Callable f;
};

template <typename Callable> UniformTransform<Callable> transform_both(Callable &&f)
{
  return UniformTransform<Callable>{std::forward<Callable>(f)};
}

// TODO Implement transform_both(f, g)
// TODO Implement transform_first(f)
// TODO Implement transform_second(f)

} // namespace bino

#endif // __BINO_H__
