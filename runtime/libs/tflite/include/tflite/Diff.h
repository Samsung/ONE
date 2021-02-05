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
 * @file     Diff.h
 * @brief    This file contains classes for testing correctess of implementation
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_DIFF_H__
#define __NNFW_TFLITE_DIFF_H__

#include "tensorflow/lite/interpreter.h"

#include "misc/RandomGenerator.h"
#include "misc/tensor/Index.h"
#include "misc/tensor/Diff.h"
#include "misc/tensor/Shape.h"
#include "misc/tensor/Comparator.h"

#include "tflite/TensorView.h"

#include <functional>
#include <vector>

/**
 * @brief Class to define TfLite interpreter match application
 */
class TfLiteInterpMatchApp
{
public:
  /**
   * @brief Construct a new TfLiteInterpMatchApp object with Comparator
   * @param[in] comparator   Comparator object for float tensor comparation
   */
  TfLiteInterpMatchApp(const nnfw::misc::tensor::Comparator<float> &comparator)
    : _verbose{false}, _comparator(comparator)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get reference verbose for debugging information
   * @return Reference of verbose value
   */
  int &verbose(void) { return _verbose; }

private:
  int _verbose;

public:
  /**
   * @brief Run two interpreter and return the output matching
   * @param[in] pure   Interpreter object of expected(with TfLite)
   * @param[in] nnapi  Interpreter object of obtained(through NNAPI)
   * @return  @c true if two Interpreter results are same, otherwise @c false
   */
  bool run(::tflite::Interpreter &pure, ::tflite::Interpreter &nnapi) const;
  /**
   * @brief Compare two Tensor values and return the match result
   * @param[in] expected  Tensor reader object to read expected values
   * @param[in] obtained  Tensor reader object to read obtained values
   * @param[in] id        Tensor ID value used for debug message
   * @return  @c true if two TensorView values are same, otherwise @c false
   */
  template <typename T>
  bool compareSingleTensorView(const nnfw::misc::tensor::Reader<T> &expected,
                               const nnfw::misc::tensor::Reader<T> &obtained, int id) const;

private:
  const nnfw::misc::tensor::Comparator<float> &_comparator;
};

#endif // __NNFW_TFLITE_DIFF_H__
