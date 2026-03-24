/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_COMMON_H_
#define _MIR_COMMON_H_

#include <cstddef>
#include <cstdint>

namespace mir
{
/**
 * @brief maximum number of dimensions what an Index, Shape or Tensor can have
 */
constexpr std::size_t MAX_DIMENSION_COUNT = 8;

inline constexpr std::size_t wrap_index(std::int32_t index, std::size_t limit) noexcept
{
  return static_cast<std::size_t>(index >= 0 ? index : limit + index);
}
} // namespace mir

#endif //_MIR_COMMON_H_
