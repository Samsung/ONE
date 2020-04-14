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
 * @file    MatrixSource.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines MatrixSource class
 */
#ifndef __INTERNAL_MATRIX_SOURCE_H__
#define __INTERNAL_MATRIX_SOURCE_H__

#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/core/Helpers.h>

#include "internal/Source.h"

/**
 * @brief Class to push matrix data to arm compute tensor
 */
template <typename T> class MatrixSource final : public Source
{
public:
  /**
   * @brief Construct a new MatrixSource object
   * @param[in] shape Shape of matrix
   * @param[in] base  Pointer of matrix data to push
   * @param[in] size  Size of matrix
   */
  MatrixSource(const nnfw::misc::matrix::Shape &shape, const T *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // do nothing
  }

public:
  /**
   * @brief       Push matrix data to arm compute tensor
   * @param[out]  tensor  Tensor object of arm compute to push matrix data
   * @return      N/A
   */
  void push(::arm_compute::ITensor &tensor) const override
  {
    using ::arm_compute::Coordinates;
    using ::arm_compute::execute_window_loop;
    using ::arm_compute::Iterator;
    using ::arm_compute::Window;

    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape(), ::arm_compute::Window::DimY);

    int32_t width = _shape.W;

    Iterator it(&tensor, window);
    execute_window_loop(window,
                        [&](const ::arm_compute::Coordinates &id) {
                          const auto height = id.y();
                          memcpy(it.ptr(), _base + height * width, width * sizeof(T));
                        },
                        it);
  }

private:
  const nnfw::misc::matrix::Shape _shape;
  const T *const _base;
  const size_t _size;
};

#endif // __INTERNAL_MATRIX_SOURCE_H__
