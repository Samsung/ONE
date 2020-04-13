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

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <cstdlib>

template <typename T> inline T *make_alloc(void)
{
  auto ptr = malloc(sizeof(T));

  if (ptr == nullptr)
  {
    throw std::bad_alloc{};
  }

  return reinterpret_cast<T *>(ptr);
}

#endif // __MEMORY_H__
