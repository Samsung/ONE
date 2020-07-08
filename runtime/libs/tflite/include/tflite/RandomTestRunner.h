/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file  RandomTestRunner.h
 * @brief This file contains class for random input testing
 */

#ifndef __NNFW_TFLITE_RANDOM_TEST_RUNNER_H__
#define __NNFW_TFLITE_RANDOM_TEST_RUNNER_H__

#include "tflite/interp/Builder.h"

#include <misc/RandomGenerator.h>

namespace nnfw
{
namespace tflite
{

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
   * @brief     Construct a new RandomTestRunner object
   * @param[in] seed          Random seed value
   * @param[in] param         RandomTestParam object for test runner
   * @param[in] quantization  TfLiteQuantizationParams type to represent quantization value
   */
  RandomTestRunner(uint32_t seed, const RandomTestParam &param)
      : _randgen{seed, 0.0f, 2.0f}, _param{param}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief     Run the random test runner
   * @param[in] running_count  Count to run tflite interpreter with NNAPI
   * @return    0 if test succeeds, otherwise failure
   */
  int run(size_t running_count);

public:
  /**
   * @brief  Get RandomGenerator reference
   * @return RandomGenerator reference
   */
  nnfw::misc::RandomGenerator &generator() { return _randgen; };

public:
  /**
   * @brief     Compile the random test runner
   * @param[in] builder  Interpreter Builder used to run
   */
  void compile(const nnfw::tflite::Builder &builder);

private:
  nnfw::misc::RandomGenerator _randgen;
  const RandomTestParam _param;
  std::unique_ptr<::tflite::Interpreter> _tfl_interp;
  std::unique_ptr<::tflite::Interpreter> _nnapi;

public:
  /**
   * @brief     Create a RandomTestRunner object
   * @param[in] seed  Random seed value
   * @return    RandomGenerator object
   */
  static RandomTestRunner make(uint32_t seed);
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_RANDOM_TEST_RUNNER_H__
