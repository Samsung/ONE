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
 * @file     TensorView.h
 * @brief    This file contains TensorView class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_TENSOR_VIEW_H__
#define __NNFW_TFLITE_TENSOR_VIEW_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"
#include "misc/tensor/Reader.h"
#include "misc/tensor/NonIncreasingStride.h"

#include <tensorflow/lite/c/c_api.h>

namespace nnfw
{
namespace tflite
{

/**
 * @brief Class to define TensorView which is inherited from nnfw::misc::tensor::Reader<T> class
 */
template <typename T> class TensorView final : public nnfw::misc::tensor::Reader<T>
{
public:
  /**
   * @brief Construct a TensorView object with base and shape informations
   * @param[in] shape The shape of a tensor
   * @param[in] base The base address of a tensor
   */
  TensorView(const nnfw::misc::tensor::Shape &shape, T *base) : _shape{shape}, _base{base}
  {
    // Set 'stride'
    _stride.init(_shape);
  }

public:
  /**
   * @brief Get shape of tensor
   * @return Reference of shape
   */
  const nnfw::misc::tensor::Shape &shape(void) const { return _shape; }

public:
  /**
   * @brief Get value of tensor index
   * @param[in] index The tensor index
   * @return The value at the index
   */
  T at(const nnfw::misc::tensor::Index &index) const override
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

public:
  /**
   * @brief Get reference value of tensor index
   * @param[in] index The tensor index
   * @return The reference value at the index
   */
  T &at(const nnfw::misc::tensor::Index &index)
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

private:
  nnfw::misc::tensor::Shape _shape; /**< The tensor shape */

public:
  T *_base;                                        /**< The base address of tensor */
  nnfw::misc::tensor::NonIncreasingStride _stride; /**< The NonIncreasingStride object */

public:
  // TODO Introduce Operand ID class
  /**
   * @brief Create TensorView object using given parameters
   * @param[in] interp The TfLite interpreter
   * @param[in] tensor_index The tensor index
   * @return The new TensorView<T> object
   */
  static TensorView<T> make(const TfLiteTensor *tensor)
  {
    // Set 'shape'
    nnfw::misc::tensor::Shape shape(TfLiteTensorNumDims(tensor));

    for (uint32_t axis = 0; axis < shape.rank(); ++axis)
    {
      shape.dim(axis) = TfLiteTensorDim(tensor, axis);
    }

    return TensorView<T>(shape, reinterpret_cast<T *>(TfLiteTensorData(tensor)));
  }
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_TENSOR_VIEW_H__
