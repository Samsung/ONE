/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  RandomGenerator.h
 * @brief This file contains classes for random value generation
 */

#ifndef __NNFW_BENCHMARK_RANDOM_GENERATOR_H__
#define __NNFW_BENCHMARK_RANDOM_GENERATOR_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"

#include <random>

namespace benchmark
{

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
  RandomGenerator(uint32_t seed, float mean, float stddev) : _rand{seed}, _dist{mean, stddev}
  {
    // DO NOTHING
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
};

template <> int8_t RandomGenerator::generate<int8_t>(void);
template <> uint8_t RandomGenerator::generate<uint8_t>(void);
template <> bool RandomGenerator::generate<bool>(void);
template <> int32_t RandomGenerator::generate<int32_t>(void);
template <> int64_t RandomGenerator::generate<int64_t>(void);

} // namespace benchmark

#endif // __NNFW_BENCHMARK_RANDOM_GENERATOR_H__
