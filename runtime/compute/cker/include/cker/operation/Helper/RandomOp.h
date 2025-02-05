/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_HELPER_RANDOM_OP_H__
#define __NNFW_CKER_HELPER_RANDOM_OP_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/operation/Helper/RandomDistributions.h"

namespace nnfw
{
namespace cker
{

namespace functor
{

template <typename Device, class Distribution> struct FillPhiloxRandom;

typedef Eigen::ThreadPoolDevice CPUDevice;
// Declares the partially CPU-specialized functor struct.
//
// NOTE: Due to inlining done by the compiler, you may need to add
// explicit instantiation of the functor in random_op.cc.  See example
// functor::FillPhiloxRandom<CPUDevice, random::UniformDistribution>.
template <class Distribution> struct FillPhiloxRandom<CPUDevice, Distribution>
{
  void operator()(random::PhiloxRandom gen, typename Distribution::ResultElementType *data,
                  int64_t size, Distribution dist);
};

} // namespace functor
} // namespace cker
} // namespace nnfw
#endif // __NNFW_CKER_HELPER_RANDOM_OP_H__
