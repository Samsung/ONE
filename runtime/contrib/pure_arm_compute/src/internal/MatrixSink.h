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
 * @file    MatrixSink.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines MatrixSink class
 */
#ifndef __INTERNAL_MATRIX_SINK_H__
#define __INTERNAL_MATRIX_SINK_H__

#include "internal/Sink.h"

#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/core/Helpers.h>

#include <cstdint>
#include <cstring>
#include <cassert>

/**
 * @brief Class to get matrix data from arm compute tensor
 */
template <typename T> class MatrixSink final : public Sink
{
public:
  /**
   * @brief Construct a new Matrix Sink object
   * @param[in] H     Height of matrix
   * @param[in] W     Width of matrix
   * @param[in] base  Pointer to get data
   * @param[in] size  Size of matrix
   */
  MatrixSink(const int32_t H, const int32_t W, T *base, const size_t size)
      : _height{H}, _width{W}, _base{base}
  {
    assert(size >= _height * _width * sizeof(T));
  }

public:
  /**
   * @brief     Get matrix data from arm compute tensor to base
   * @param[in] tensor  Tensor object of arm compute to get data
   * @return    N/A
   */
  void pull(::arm_compute::ITensor &tensor) const override
  {
    assert(tensor.info()->dimension(0) == _width);
    assert(tensor.info()->dimension(1) == _height);

    using ::arm_compute::Coordinates;
    using ::arm_compute::execute_window_loop;
    using ::arm_compute::Iterator;
    using ::arm_compute::Window;

    Window window;

    window.use_tensor_dimensions(tensor.info()->tensor_shape(), ::arm_compute::Window::DimY);

    Iterator it(&tensor, window);
    execute_window_loop(window,
                        [&](const ::arm_compute::Coordinates &id) {
                          const auto row = id.y();
                          memcpy(_base + row * _width, it.ptr(), _width * sizeof(T));
                        },
                        it);
  }

private:
  const int32_t _height;
  const int32_t _width;

private:
  T *const _base;
};

#endif // __INTERNAL_MATRIX_SINK_H__
