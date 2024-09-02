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

#ifndef __ONERT_EXEC_FEATURE_NHWC_READER_H__
#define __ONERT_EXEC_FEATURE_NHWC_READER_H__

#include "../Reader.h"

#include <cassert>

#include "backend/ITensor.h"
#include "ir/Shape.h"
#include "util/Utils.h"

namespace onert
{
namespace exec
{
namespace feature
{
namespace nhwc
{

template <typename T> class Reader : public feature::Reader<T>
{
public:
  using Strides = ir::FeatureShape;
  // Construct for buffer and strides
  Reader(const ir::FeatureShape &shape, const Strides &strides, const T *ptr, size_t len)
    : _shape{shape}, _strides{strides}, _ptr{reinterpret_cast<const uint8_t *>(ptr)}, _len{len}
  {
    UNUSED_RELEASE(len); // Workaround for unused variable in release mode
    assert(len == static_cast<size_t>(strides.N != 0   ? shape.N * strides.N
                                      : strides.H != 0 ? shape.H * strides.H
                                      : strides.W != 0 ? shape.W * strides.W
                                                       : shape.C * strides.C));
  }

  // Construct for backend tensor
  Reader(const backend::ITensor *tensor)
    : _ptr{tensor->buffer() + tensor->calcOffset({0, 0, 0, 0})}, _len{tensor->total_size()}
  {
    const auto start_offset = tensor->calcOffset({0, 0, 0, 0});
    auto shape = tensor->getShape();
    _strides.C = shape.dim(3) == 1 ? 0 : tensor->calcOffset({0, 0, 0, 1}) - start_offset;
    _strides.W = shape.dim(2) == 1 ? 0 : tensor->calcOffset({0, 0, 1, 0}) - start_offset;
    _strides.H = shape.dim(1) == 1 ? 0 : tensor->calcOffset({0, 1, 0, 0}) - start_offset;
    _strides.N = shape.dim(0) == 1 ? 0 : tensor->calcOffset({1, 0, 0, 0}) - start_offset;

    _shape.C = shape.dim(3);
    _shape.W = shape.dim(2);
    _shape.H = shape.dim(1);
    _shape.N = shape.dim(0);
  }

public:
  T at(uint32_t batch, uint32_t row, uint32_t col, uint32_t ch) const final
  {
    return getRef(batch, row, col, ch);
  }
  T at(uint32_t row, uint32_t col, uint32_t ch) const final { return getRef(0, row, col, ch); }

protected:
  const T &getRef(uint32_t batch, uint32_t row, uint32_t col, uint32_t ch) const
  {
    const auto offset = feature_index_to_byte_offset(batch, row, col, ch);

    const T *ptr = reinterpret_cast<const T *>(_ptr + offset);

    return *ptr;
  }

private:
  size_t feature_index_to_byte_offset(uint32_t batch, uint32_t row, uint32_t col, uint32_t ch) const
  {
    assert(1u * _shape.N > batch); // shape.N > batch
    assert(1u * _shape.H > row);   // shape.H > row
    assert(1u * _shape.W > col);   // shape.W > col
    assert(1u * _shape.C > ch);    // shape.C > ch

    uint32_t res = 0;
    res += batch * _strides.N;
    res += row * _strides.H;
    res += col * _strides.W;
    res += ch * _strides.C;

    return res;
  }

private:
  // TODO Remove _shape
  ir::FeatureShape _shape;
  Strides _strides;
  const uint8_t *_ptr;
  size_t _len;
};

} // namespace nhwc
} // namespace feature
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FEATURE_NHWC_READER_H__
