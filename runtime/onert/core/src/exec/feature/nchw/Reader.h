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

#ifndef __ONERT_EXEC_FEATURE_NCHW_READER_H__
#define __ONERT_EXEC_FEATURE_NCHW_READER_H__

#include "../Reader.h"

#include <cassert>

#include "backend/ITensor.h"
#include "misc/feature/Shape.h"

namespace onert
{
namespace exec
{
namespace feature
{
namespace nchw
{

template <typename T> class Reader final : public feature::Reader<T>
{
public:
  // Construct for buffer of model inputs
  Reader(const ::nnfw::misc::feature::Shape &shape, const T *ptr, size_t len)
      : _shape{shape}, _ptr{reinterpret_cast<const uint8_t *>(ptr)}, _len{len}
  {
    assert(shape.N * shape.C * shape.H * shape.W * sizeof(T) == len);

    // No padding
    _strides.W = sizeof(T);
    _strides.H = shape.W * sizeof(T);
    _strides.C = shape.W * shape.H * sizeof(T);
    _strides.N = shape.W * shape.H * shape.C * sizeof(T);
  }

  // Construct for backend tensor
  Reader(backend::ITensor *tensor)
      : _ptr{tensor->buffer() + tensor->calcOffset({0, 0, 0, 0})}, _len{tensor->total_size()}
  {
    assert(tensor->layout() == ir::Layout::NCHW);

    const auto start_offset = tensor->calcOffset({0, 0, 0, 0});
    _strides.W = tensor->dimension(3) == 1 ? 0 : tensor->calcOffset({0, 0, 0, 1}) - start_offset;
    _strides.H = tensor->dimension(2) == 1 ? 0 : tensor->calcOffset({0, 0, 1, 0}) - start_offset;
    _strides.C = tensor->dimension(1) == 1 ? 0 : tensor->calcOffset({0, 1, 0, 0}) - start_offset;
    _strides.N = tensor->dimension(0) == 1 ? 0 : tensor->calcOffset({1, 0, 0, 0}) - start_offset;

    _shape.W = tensor->dimension(3);
    _shape.H = tensor->dimension(2);
    _shape.C = tensor->dimension(1);
    _shape.N = tensor->dimension(0);
  }

public:
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    const auto offset = feature_index_to_byte_offset(0, ch, row, col);

    const T *ptr = reinterpret_cast<const T *>(_ptr + offset);

    return *ptr;
  }
  T at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    const auto offset = feature_index_to_byte_offset(batch, ch, row, col);

    const T *ptr = reinterpret_cast<const T *>(_ptr + offset);

    return *ptr;
  }

private:
  size_t feature_index_to_byte_offset(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const
  {
    assert(1u * _shape.N > batch); // shape.N > batch
    assert(1u * _shape.C > ch);    // shape.C > ch
    assert(1u * _shape.H > row);   // shape.H > row
    assert(1u * _shape.W > col);   // shape.W > col

    uint32_t res = 0;
    res += batch * _strides.N;
    res += ch * _strides.C;
    res += row * _strides.H;
    res += col * _strides.W;

    return res;
  }

private:
  // TODO Remove _shape
  nnfw::misc::feature::Shape _shape;
  using Strides = nnfw::misc::feature::Shape;
  Strides _strides;
  const uint8_t *_ptr;
  size_t _len;
};

} // namespace nchw
} // namespace feature
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FEATURE_NCHW_READER_H__
