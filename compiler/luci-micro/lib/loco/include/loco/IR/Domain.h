/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_DOMAIN_H__
#define __LOCO_IR_DOMAIN_H__

namespace loco
{

/**
 * @brief Describe the kind of (N-dimensional) loco values
 *
 * loco is an intermediate representation for neural network compiler, which mainly focuses on
 * N-dimensional values (usually referred to as Tensor).
 *
 * There are several special cases for N-dimensional values according to its usage. For example,
 * vision community often refers to 4D array as "FeatureMap".
 *
 * It is definitely possible to represent all of these special cases using Tensor, but that scheme
 * may introduces some confusion (e.g. NCHW vs NHWC issue).
 *
 * loco distinguishes these special cases from Tensor in order to reduce such confusion.
 *
 * This "Domain" enum class enumerates all of these special cases that loco supports.
 */
enum class Domain
{
  Unknown,
  Tensor,
  Feature,
  Filter,          /* 2D Convolution Filter */
  DepthwiseFilter, /* Depthwise 2D Convolution Filter */
  Bias,
  Matrix,
  /* ... */
};

} // namespace loco

#endif // __LOCO_IR_DOMAIN_H__
