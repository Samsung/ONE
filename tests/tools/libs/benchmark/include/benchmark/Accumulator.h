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
 * @file benchmark.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains benchmark::Accumulator class
 */
#ifndef __NNFW_BENCHMARK_ACCUMULATOR_H__
#define __NNFW_BENCHMARK_ACCUMULATOR_H__

#include <chrono>

// Benckmark support
namespace benchmark
{

/**
 * @brief Class to accumulate time during benchmark
 */
template <typename T> class Accumulator
{
public:
  /**
   * @brief Construct a new Accumulator object
   * @param[in] ref   Object to keep time duration
   */
  Accumulator(T &ref) : _ref(ref)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Return the reference of @c ref passed to constructor
   * @return Reference of @c ref
   */
  T &operator()(void) { return _ref; }

private:
  T &_ref;
};

/**
 * @brief Run passed function and returns accumulated time
 * @tparam T            Period used by @c std::chrono::duration_cast
 * @tparam Callable     Function type to benchmark
 * @param[in] acc       Accumulated time after running @cb
 * @param[in] cb        Function to run and benchmark
 * @return Accumulated time
 */
template <typename T, typename Callable>
Accumulator<T> &operator<<(Accumulator<T> &&acc, Callable cb)
{
  auto begin = std::chrono::high_resolution_clock::now();
  cb();
  auto end = std::chrono::high_resolution_clock::now();

  acc() += std::chrono::duration_cast<T>(end - begin);

  return acc;
}

template <typename T> Accumulator<T> measure(T &out) { return Accumulator<T>(out); }

} // namespace benchmark

#endif // __NNFW_BENCHMARK_ACCUMULATOR_H__
