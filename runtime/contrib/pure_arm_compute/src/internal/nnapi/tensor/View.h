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
 * @file    View.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines internal::nnapi::tensor::View class
 */
#ifndef __INTERNAL_NNAPI_TENSOR_VIEW_H__
#define __INTERNAL_NNAPI_TENSOR_VIEW_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"

namespace internal
{
namespace nnapi
{
namespace tensor
{

/**
 * @brief Class to access tensor's element information using index
 */
template <typename T> class View
{
public:
  /**
   * @brief     Construct a new View object
   * @param[in] shape Shape of tensor
   * @param[in] ptr   Pointer to tensor data
   * @param[in] len   Size of tensor (byte)
   */
  // NOTE The parameter len denotes the number of bytes.
  View(const ::nnfw::misc::tensor::Shape &shape, T *ptr, size_t len) : _shape{shape}, _ptr{ptr}
  {
    assert(shape.num_elements() * sizeof(T) == len);
  }

public:
  /**
   * @brief   Get shape of tensor
   * @return  Shape of tensor
   */
  const nnfw::misc::tensor::Shape &shape(void) const { return _shape; }

private:
  /**
   * @brief     Get position of element using index in tensor
   * @param[in] index Index of element
   * @return    Position of element
   */
  uint32_t offset_of(const nnfw::misc::tensor::Index &index) const
  {
    if (_shape.rank() == 0)
    {
      return 0;
    }

    uint32_t offset = index.at(0);

    // Stride decreases as axis increases in NNAPI
    for (uint32_t axis = 1; axis < _shape.rank(); ++axis)
    {
      offset *= _shape.dim(axis);
      offset += index.at(axis);
    }

    return offset;
  }

public:
  /**
   * @brief     Get value of element at index
   * @param[in] index Index of element
   * @return    Value of element at index
   */
  T at(const nnfw::misc::tensor::Index &index) const
  {
    const auto offset = offset_of(index);

    return _ptr[offset];
  }

  /**
   * @brief     Get reference of element at index
   * @param[in] index Index of element
   * @return    Reference of element at index
   */
  T &at(const nnfw::misc::tensor::Index &index)
  {
    const auto offset = offset_of(index);

    return _ptr[offset];
  }

private:
  nnfw::misc::tensor::Shape _shape;

private:
  T *_ptr;
};

} // namespace tensor
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_TENSOR_VIEW_H__
