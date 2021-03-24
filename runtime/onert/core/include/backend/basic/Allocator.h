/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file        Allocator.h
 * @brief       This file contains Allocator related classes
 */

#ifndef __ONERT_BACKEND_BASIC_ALLOCATOR_H__
#define __ONERT_BACKEND_BASIC_ALLOCATOR_H__

#include <memory>

namespace onert
{
namespace backend
{
namespace cpu_common
{

/**
 * @brief Class to allocate memory
 */
class Allocator
{
public:
  Allocator(uint32_t capacity);
  /**
   * @brief Get memory base pointer
   * @return base pointer
   */
  uint8_t *base() const { return _base.get(); }
  void release() { _base.reset(); }

private:
  std::unique_ptr<uint8_t[]> _base;
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_ALLOCATOR_H__
