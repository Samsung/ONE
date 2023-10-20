/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MINMAX_EMBEDDER_TEST_CAST_H__
#define __MINMAX_EMBEDDER_TEST_CAST_H__

#include <cstdint>
#include <stdexcept>

namespace minmax_embedder_test
{
uint32_t to_u32(uint64_t v)
{
  if (v > UINT32_MAX)
    throw std::overflow_error("to_u32 gets a value bigger than uint32 max.");
  return static_cast<uint32_t>(v);
}

} // end of namespace minmax_embedder_test

#endif // __MINMAX_EMBEDDER_TEST_CAST_H__
