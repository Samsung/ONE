/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_MISC_TO_UNDERLYING_H__
#define __NNFW_MISC_TO_UNDERLYING_H__

#include <type_traits>

namespace nnfw
{
namespace misc
{

template <typename E> constexpr auto to_underlying(E e) noexcept
{
  return static_cast<std::underlying_type_t<E>>(e);
}

} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TO_UNDERLYING_H__
