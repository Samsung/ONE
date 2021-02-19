/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_TFLITE_COMPARATOR_MATCH_APP_H__
#define __NNFW_TFLITE_COMPARATOR_MATCH_APP_H__

#include "IOManager.h"

#include <misc/tensor/Comparator.h>
#include <tensorflow/lite/interpreter.h>

namespace nnfw
{
namespace onert_cmp
{

/**
 * @brief Class to define TfLite interpreter match application
 */
class MatchApp
{
public:
  /**
   * @brief Construct a new MatchApp object with Comparator
   * @param[in] comparator   Comparator object for float tensor comparation
   */
  MatchApp(const nnfw::misc::tensor::Comparator &comparator)
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

  /**
   * @brief Run two interpreter and return the output matching
   * @param[in] tflite   TFLite Interpreter object of expected
   * @param[in] manager  nnfw API IO Manager object of obtained
   * @return  @c true if two Interpreter results are same, otherwise @c false
   */
  bool run(::tflite::Interpreter &tflite, IOManager &manager) const;

private:
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
  int _verbose;
  const nnfw::misc::tensor::Comparator &_comparator;
};

} // namespace onert_cmp
} // namespace nnfw

#endif // __NNFW_TFLITE_COMPARATOR_MATCH_APP_H__
