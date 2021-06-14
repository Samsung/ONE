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

#ifndef __ONERT_BACKEND_GPU_CL_MODEL_HINTS_H__
#define __ONERT_BACKEND_GPU_CL_MODEL_HINTS_H__

#include <cstdint>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct ModelHints
{
  using ModelHint = uint64_t;

  // By default we want the fastest inference.
  static constexpr ModelHint kFastestInference = 0x00000000;
  // Can improve compilation time, but inference can be slower.
  static constexpr ModelHint kReduceKernelsCount = 0x00000001;
  // Can improve tuning time, but inference can be slower.
  static constexpr ModelHint kFastTuning = 0x00000002;

  // Experimental.
  // Can improve performance and memory consumption, but slow down
  // initialization a lot and create more kernels.
  static constexpr ModelHint kAllowSpecialKernels = 0x00000004;

  void Add(ModelHint hint)
  {
    if (hint == kFastestInference)
    {
      hints = kFastestInference;
    }
    else
    {
      hints |= hint;
    }
  }

  bool Check(ModelHint hint) const { return hints & hint; }

  uint64_t hints = kFastestInference;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_MODEL_HINTS_H__
