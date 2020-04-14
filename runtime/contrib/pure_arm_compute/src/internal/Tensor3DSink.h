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
 * @file    Tensor3DSink.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines Tensor3DSink class
 */
#ifndef __TENSOR3D_SINK_H__
#define __TENSOR3D_SINK_H__

#include "internal/Sink.h"

//
// This is mempcy() version of generic TensorSink for 3D tensor
//
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/core/Helpers.h>

/**
 * @brief Class to get tensor data from arm compute tensor
 */
template <typename T> class Tensor3DSink final : public Sink
{
public:
  /**
   * @brief     Construct a new Tensor3DSink object
   * @param[in] shape Shape of tensor
   * @param[in] base  Pointer to get data
   * @param[in] size  Size of tensor
   */
  Tensor3DSink(const nnfw::misc::tensor::Shape &shape, T *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief     Get tensor data from arm compute tensor to base
   * @param[in] tensor  Tensor object of arm compute to get data
   * @return    N/A
   */
  void pull(::arm_compute::ITensor &tensor) const override
  {
    using ::arm_compute::Coordinates;
    using ::arm_compute::execute_window_loop;
    using ::arm_compute::Iterator;
    using ::arm_compute::Window;

    Window window;

    window.use_tensor_dimensions(tensor.info()->tensor_shape(), ::arm_compute::Window::DimY);
    int32_t height_width = _shape.dim(1) * _shape.dim(2);
    int32_t width = _shape.dim(2);

    Iterator it(&tensor, window);
    execute_window_loop(window,
                        [&](const ::arm_compute::Coordinates &id) {
                          const auto z = id.z();
                          const auto y = id.y();
                          memcpy(_base + z * height_width + y * width, it.ptr(), width * sizeof(T));
                        },
                        it);
  }

private:
  const nnfw::misc::tensor::Shape _shape;

private:
  T *const _base;
  const size_t _size;
};

#endif // __TENSOR3D_SINK_H__
