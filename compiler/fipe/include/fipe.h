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

#ifndef __FIPE_H__
#define __FIPE_H__

#include <functional>
#include <utility>

namespace fipe
{

/**
 * @brief Convert a function pointer as a callable std::function
 *
 * NOTE "fipe" works only for unary functions.
 */
template <typename Ret, typename Arg> std::function<Ret(Arg)> wrap(Ret (*p)(Arg)) { return p; }

} // namespace fipe

template <typename T, typename Callable> auto operator|(T &&v, Callable &&f) -> decltype(f(v))
{
  return std::forward<Callable>(f)(v);
}

#endif // __FIPE_H__
