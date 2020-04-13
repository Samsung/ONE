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

/**
 * @file execution.h
 * @brief This file defines ANeuralNetworksMemory class for handling Memory NNAPI
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <cstdint>

/**
 * @brief struct to define Memory NNAPI
 */
struct ANeuralNetworksMemory
{
public:
  /**
   * @brief Constructor with params
   * @param [in] size The requested size in bytes
   * @param [in] protect The desired memory protection for the mapping
   * @param [in] fd The requested file descriptor
   * @param [in] offset The offset to the beginning of the file of the area to map
   */
  ANeuralNetworksMemory(size_t size, int protect, int fd, size_t offset);
  /**
   * @brief Destructor default
   */
  ~ANeuralNetworksMemory();

public:
  /**
   * @brief Get size
   * @return size
   */
  size_t size(void) const { return _size; }
  /**
   * @brief Get base pointer
   * @return base pointer
   */
  uint8_t *base(void) { return _base; }
  /**
   * @brief Get base pointer
   * @return const base pointer
   */
  const uint8_t *base(void) const { return _base; }

private:
  size_t _size;
  uint8_t *_base;
};

#endif // __MEMORY_H__
