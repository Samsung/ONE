/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_HELPER_RANDOM_OP_CPU_H__
#define __NNFW_CKER_HELPER_RANDOM_OP_CPU_H__

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <memory>

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/eigen/EigenSupport.h"

#include "cker/operation/Helper/PhiloxRandom.h"
#include "cker/operation/Helper/RandomOp.h"
#include "cker/operation/Helper/RandomDistributions.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace nnfw
{
namespace cker
{

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor
{
using random::PhiloxRandom;
using random::SingleSampleAdapter;

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution> struct FillPhiloxRandom
{
  typedef typename Distribution::ResultElementType T;
  void operator()() {}
};

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput> struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution> struct FillPhiloxRandomTask<Distribution, false>
{
  typedef typename Distribution::ResultElementType T;
  static void Run(random::PhiloxRandom gen, T *data, int64_t size, Distribution dist)
  {
    const int kGroupSize = Distribution::kResultElementCount;
    gen.Skip(0);
    int64_t offset = 0;

    // First fill all the full-size groups
    int64_t limit_group_full = size / kGroupSize;
    for (int64_t index = 0; index < limit_group_full; ++index)
    {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    int64_t remaining_size = size - limit_group_full * kGroupSize;

    // If there are any remaining elements that need to be filled, process them
    if (remaining_size > 0)
    {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution> struct FillPhiloxRandomTask<Distribution, true>
{
  typedef typename Distribution::ResultElementType T;
  static constexpr int64_t kReservedSamplesPerOutput = 256;

  static void Run(random::PhiloxRandom base_gen, T *data, int64_t size, Distribution dist)
  {
    const int kGroupSize = Distribution::kResultElementCount;
    static const int kGeneratorSkipPerOutputGroup =
      kGroupSize * kReservedSamplesPerOutput / PhiloxRandom::kResultElementCount;

    int64_t offset = 0;

    // First fill all the full-size groups
    int64_t limit_group_full = size / kGroupSize;
    int64_t group_index;
    for (group_index = 0; group_index < limit_group_full; ++group_index)
    {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    int64_t remaining_size = size - limit_group_full * kGroupSize;
    // If there are any remaining elements that need to be filled, process them
    if (remaining_size > 0)
    {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
void FillPhiloxRandom<CPUDevice, Distribution>::operator()(
  random::PhiloxRandom gen, typename Distribution::ResultElementType *data, int64_t size,
  Distribution dist)
{
  FillPhiloxRandomTask<Distribution, Distribution::kVariableSamplesPerOutput>::Run(gen, data, size,
                                                                                   dist);
}

} // namespace functor

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_HELPER_RANDOM_OP_CPU_H__
