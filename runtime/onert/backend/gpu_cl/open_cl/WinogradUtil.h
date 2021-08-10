/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_WINOGRAD_UTIL_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_WINOGRAD_UTIL_H__

#include <vector>

#include "open_cl/DataType.h"
#include "open_cl/Shape.h"
#include "open_cl/InternalTensor.h"

namespace onert
{
namespace backend
{

// Matrices for Winograd trasformations received with method described here
// https://openreview.net/pdf?id=H1ZaRZVKg

// returns A transposed matrix(6 * 4) as array (24 values) for Winograd4x4To6x6
std::vector<float> AtMatrixForWinograd4x4To6x6();

// returns B transposed matrix(6 * 6) as array (36 values) for Winograd4x4To6x6
std::vector<float> BtMatrixForWinograd4x4To6x6();

void RearrangeWeightsToWinograd4x4To6x6Weights(
  const gpu_cl::InternalTensor<gpu_cl::OHWI, gpu_cl::DataType::FLOAT32> &src_weights,
  gpu_cl::InternalTensor<gpu_cl::OHWI, gpu_cl::DataType::FLOAT32> *dst_weights);

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_WINOGRAD_UTIL_H__
