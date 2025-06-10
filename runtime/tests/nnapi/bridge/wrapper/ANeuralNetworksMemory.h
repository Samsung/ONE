/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <cstdint>

struct ANeuralNetworksMemory
{
public:
  ANeuralNetworksMemory(size_t size, int protect, int fd, size_t offset);
  ~ANeuralNetworksMemory();

public:
  size_t size(void) const { return _size; }
  uint8_t *base(void) { return _base; }
  uint8_t *base(void) const { return _base; }
  bool vaildAccess(size_t offset, size_t length) const;

private:
  size_t _size;
  uint8_t *_base;
};

#endif // __MEMORY_H__
