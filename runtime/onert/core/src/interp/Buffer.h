/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file  Buffer.h
 * @brief This file contains Buffer interface and InternalBuffer, ExternalBuffer class
 */
#ifndef __ONERT_INTERP_BUFFER_H__
#define __ONERT_INTERP_BUFFER_H__

#include <memory>

#include "ir/Data.h"

namespace onert
{
namespace interp
{

/**
 * @brief Interface for writable data area
 */
class Buffer : public ir::Data
{
public:
  /**
   * @brief   Return writable pointer for data area
   * @return  Writable pointer
   */
  virtual uint8_t *baseWritable(void) const = 0;
};

/**
 * @brief Class for internally allocated data area
 */
class InternalBuffer final : public Buffer
{
public:
  InternalBuffer(size_t size) : _base{std::make_unique<uint8_t[]>(size)}, _size{size}
  {
    // DO NOTHING
  }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base.get(); }
  uint8_t *baseWritable(void) const override { return _base.get(); }

private:
  std::unique_ptr<uint8_t[]> _base;
  size_t _size;
};

/**
 * @brief Class for data area from outside
 */
class ExternalBuffer final : public Buffer
{
public:
  ExternalBuffer(uint8_t *base, size_t size) : _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base; }
  uint8_t *baseWritable(void) const override { return _base; }

private:
  uint8_t *_base;
  size_t _size;
};

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_BUFFER_H__
