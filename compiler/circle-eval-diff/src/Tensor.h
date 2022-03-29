/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_EVAL_DIFF_TENSOR_H__
#define __CIRCLE_EVAL_DIFF_TENSOR_H__

#include <loco.h>

#include <vector>

namespace circle_eval_diff
{

struct TensorDataType
{
public:
  const loco::DataType &dtype(void) const { return _dtype; }
  void dtype(const loco::DataType &dtype) { _dtype = dtype; }

private:
  loco::DataType _dtype = loco::DataType::Unknown;
};

struct TensorShape
{
public:
  uint32_t rank(void) const { return _dims.size(); }
  void rank(uint32_t value) { _dims.resize(value); }

  const loco::Dimension &dim(uint32_t axis) const { return _dims.at(axis); }
  loco::Dimension &dim(uint32_t axis) { return _dims.at(axis); }

  void shape(std::initializer_list<uint32_t> dims)
  {
    rank(dims.size());

    uint32_t axis = 0;
    for (auto d : dims)
    {
      dim(axis++) = d;
    }
  }

private:
  std::vector<loco::Dimension> _dims;
};

// Tensor has three kinds of data
// 1. DataType (_dtype)
// 2. Shape (_dims)
// 3. Buffer (_data)
struct Tensor final : public TensorShape, public TensorDataType
{
public:
  template <loco::DataType DT> uint32_t size(void) const;
  template <loco::DataType DT> void size(uint32_t size);
  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
  template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &at(uint32_t n);
  uint8_t *buffer(void) { return _data.data(); }
  uint32_t byte_size(void) const { return _data.size(); }

private:
  std::vector<uint8_t> _data;
};

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_TENSOR_H__
