/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_STATELESS_RANDOM_UNIFORM_H__
#define __NNFW_CKER_STATELESS_RANDOM_UNIFORM_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/eigen/EigenSupport.h"

#include "cker/operation/Helper/Tensor.h"
#include "cker/operation/Helper/PhiloxRandom.h"
#include "cker/operation/Helper/RandomOpCpu.h"
#include "cker/operation/Helper/RandomDistributions.h"

namespace nnfw
{
namespace cker
{

void GenerateKey(Tensor seed, random::PhiloxRandom::Key *out_key,
                 random::PhiloxRandom::ResultType *out_counter)
{
  // Grab the two seeds
  uint32_t seed0;
  uint32_t seed1;

  const auto seed_vals = seed.flat<int32_t>();

  seed0 = seed_vals(0);
  seed1 = seed_vals(1);
  // Scramble the seeds so that the user doesn't need to worry about which
  // part of the seed needs to be strong.
  (*out_key)[0] = 0x3ec8f720;
  (*out_key)[1] = 0x02461e29;
  (*out_counter)[0] = static_cast<uint32_t>(seed0);
  (*out_counter)[1] = (*out_counter)[3] = 0;
  (*out_counter)[2] = static_cast<uint32_t>(seed1);
  const auto mix = random::PhiloxRandom(*out_counter, *out_key)();
  (*out_key)[0] = mix[0];
  (*out_key)[1] = mix[1];
  (*out_counter)[0] = (*out_counter)[1] = 0;
  (*out_counter)[2] = mix[2];
  (*out_counter)[3] = mix[3];
}

template <typename Device, class Distribution>
void Fill(random::PhiloxRandom random, Tensor *output)
{
  // Build distribution
  typedef typename Distribution::ResultElementType T;

  auto flat = output->flat<T>();
  // Reuse the compute kernels from the stateful random ops
  functor::FillPhiloxRandom<Device, Distribution>()(random, flat.data(), flat.size(),
                                                    Distribution());
}

inline void StatelessRandomUniform(const Shape &shape_shape, const int *shape_data,
                                   const Shape &seed_shape, const int *seed_data,
                                   const Shape &output_shape, float *output_data)
{
  Tensor shape_t;
  Tensor seed_t;

  shape_t.shape.ReplaceWith(shape_shape.DimensionsCount(), shape_shape.DimsData());
  shape_t.buffer = (void *)shape_data;

  seed_t.shape.ReplaceWith(seed_shape.DimensionsCount(), seed_shape.DimsData());
  seed_t.buffer = (void *)seed_data;

  Tensor output_t;
  output_t.shape.ReplaceWith(output_shape.DimensionsCount(), output_shape.DimsData());
  output_t.buffer = output_data;

  random::PhiloxRandom::Key key;
  random::PhiloxRandom::ResultType counter;

  GenerateKey(seed_t, &key, &counter);

  Fill<Eigen::ThreadPoolDevice, random::UniformDistribution<random::PhiloxRandom, float>>(
    random::PhiloxRandom(counter, key), &output_t);
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_STATELESS_RANDOM_UNIFORM_H__
