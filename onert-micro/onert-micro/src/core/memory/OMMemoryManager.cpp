/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "core/memory/OMMemoryManager.h"

using namespace onert_micro::core::memory;
using namespace onert_micro;

OMStatus OMMemoryManager::allocateMemory(uint32_t size, uint8_t **data)
{
  if (size == 0)
    return UnknownError;
  auto data_tmp = new uint8_t[size];

  *data = data_tmp;

  if (*data == nullptr)
    return UnknownError;

  return Ok;
}

OMStatus OMMemoryManager::deallocateMemory(uint8_t *data)
{
  delete[] data;
  return Ok;
}
