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
 * @file     FeatureView.h
 * @brief    This file contains FeatureView class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_FEATURE_VIEW_H__
#define __NNFW_TFLITE_FEATURE_VIEW_H__

#include "tensorflow/lite/interpreter.h"

#include "tflite/InputIndex.h"
#include "tflite/OutputIndex.h"

#include "misc/feature/Shape.h"
#include "misc/feature/Reader.h"

namespace nnfw
{
namespace tflite
{

template <typename T> class FeatureView;

/**
 * @brief Class to support reading element of float type feature
 */
template <> class FeatureView<float> : public nnfw::misc::feature::Reader<float>
{
public:
  /**
   * @brief     Construct a new FeatureView object
   * @param[in] interp  Interpreter to read from
   * @param[in] index   InputIndex index of input
   */
  FeatureView(::tflite::Interpreter &interp, const InputIndex &index);
  /**
   * @brief     Construct a new FeatureView object
   * @param[in] interp  Interpreter to read from
   * @param[in] index   OutputIndex index of output
   */
  FeatureView(::tflite::Interpreter &interp, const OutputIndex &index);

public:
  /**
   * @brief     Get value of element using channel, row and column index
   * @param[in] ch    Channel index
   * @param[in] row   Row index
   * @param[in] col   Column index
   * @return    Value of element
   */
  float at(uint32_t ch, uint32_t row, uint32_t col) const;
  /**
   * @brief     Get reference of element using channel, row and column index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Reference of element
   */
  float &at(uint32_t ch, uint32_t row, uint32_t col);

  float at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const = 0;

private:
  /**
   * @brief     Get offset of element from channel, row and column index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Offset of element
   */
  uint32_t getElementOffset(uint32_t ch, uint32_t row, uint32_t col) const
  {
    uint32_t res = 0;

    // TensorFlow Lite assumes that NHWC ordering for tessor
    res += row * _shape.W * _shape.C;
    res += col * _shape.C;
    res += ch;

    return res;
  }

private:
  nnfw::misc::feature::Shape _shape;
  float *_base;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_FEATURE_VIEW_H__
