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

#include "backend/basic/Allocator.h"

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace basic
{

Allocator::Allocator(uint32_t capacity)
{
  _base = std::make_unique<uint8_t[]>(capacity);

  VERBOSE(ALLOC) << "allocation capacity: " << capacity << std::endl;
  VERBOSE(ALLOC) << "base pointer: " << static_cast<void *>(_base.get()) << std::endl;
}

} // namespace basic
} // namespace backend
} // namespace onert
