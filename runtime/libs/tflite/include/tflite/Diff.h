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
   * @param[in] comparator   Comparator object for tensor comparation
   */
  TfLiteInterpMatchApp(const nnfw::misc::tensor::Comparator &comparator)
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
   * @brief Compare two TensorView values and return the match result
   * @param[in] expected  TensorView object to read expected values
   * @param[in] obtained  TensorView object to read obtained values
   * @param[in] id        Tensor ID value used for debug message
   * @return  @c true if two TensorView values are same, otherwise @c false
   */
  template <typename T>
  bool compareSingleTensorView(const nnfw::tflite::TensorView<T> &expected,
                               const nnfw::tflite::TensorView<T> &obtained, int id) const;

private:
  const nnfw::misc::tensor::Comparator &_comparator;
};

#include "tflite/interp/Builder.h"
#include "tflite/Quantization.h"

#include <random>

/**
 * @brief Class to generate random values
 */
class RandomGenerator
{
public:
  /**
   * @brief Construct a new RandomGenerator object
   * @param[in] seed          Random seed value
   * @param[in] mean          Mean value of normal random number generation
   * @param[in] stddev        Standard deviation of random number generation
   * @param[in] quantization  TfLiteQuantizationParams type to represent quantization value
   *                          (not used yet)
   */
  RandomGenerator(uint32_t seed, float mean, float stddev,
                  const TfLiteQuantizationParams quantization = make_default_quantization())
      : _rand{seed}, _dist{mean, stddev}, _quantization{quantization}
  {
    (void)_quantization;
  }

public:
  /**
   * @brief  Generate random numbers for type T
   * @param[in] s Shape value
   * @param[in] i Index value
   * @return Random generated value
   * @note   This is same as T generate(void) as two input parameters are not used
   */
  template <typename T>
  T generate(const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)
  {
    return generate<T>();
  }

  /**
   * @brief  Generate random numbers for type T
   * @return Random generated value
   */
  template <typename T> T generate(void) { return _dist(_rand); }

private:
  std::minstd_rand _rand;
  std::normal_distribution<float> _dist;
  // unused
  const TfLiteQuantizationParams _quantization;
};

template <> uint8_t RandomGenerator::generate<uint8_t>(void);
template <> bool RandomGenerator::generate<bool>(void);
template <> int32_t RandomGenerator::generate<int32_t>(void);

/**
 * @brief Structure for NNAPI correctness test
 */
struct RandomTestParam
{
  int verbose;               //!< Verbosity of debug information
  int tolerance;             //!< Torlerance of value difference
  int tensor_logging = 0;    //!< Save logging to a file if not 0
  std::string log_path = ""; //!< Path of log file, meaningful only when tensor_logging is 1
};

/**
 * @brief Class to define Random test runner
 */
class RandomTestRunner
{
public:
  /**
   * @brief Construct a new RandomTestRunner object
   * @param[in] seed          Random seed value
   * @param[in] param         RandomTestParam object for test runner
   * @param[in] quantization  TfLiteQuantizationParams type to represent quantization value
   */
  RandomTestRunner(uint32_t seed, const RandomTestParam &param,
                   const TfLiteQuantizationParams quantization = make_default_quantization())
      : _randgen{seed, 0.0f, 2.0f, quantization}, _param{param}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief  Run the random test runner
   * @param[in] builder  Interpreter Builder used to run
   * @return 0 if test succeeds, otherwise failure
   */
  int run(const nnfw::tflite::Builder &builder);

public:
  /**
   * @brief  Get RandomGenerator reference
   * @return RandomGenerator reference
   */
  RandomGenerator &generator() { return _randgen; };

private:
  RandomGenerator _randgen;
  const RandomTestParam _param;

public:
  /**
   * @brief  Create a RandomTestRunner object
   * @param[in] seed  Random seed value
   * @return RandomGenerator object
   */
  static RandomTestRunner make(uint32_t seed);
};

#endif // __NNFW_TFLITE_DIFF_H__
