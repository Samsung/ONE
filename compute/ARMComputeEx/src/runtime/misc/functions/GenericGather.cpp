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

#include "arm_compute/runtime/misc/functions/GenericGather.h"

namespace arm_compute
{
namespace misc
{

bool shouldPermute(arm_compute::ITensorInfo *input, arm_compute::ITensorInfo *output)
{
  return (input->num_dimensions() != 4 && output->num_dimensions() == 4 &&
          input->data_layout() == DataLayout::NCHW);
}

void GenericGather::configure(arm_compute::ITensor *input, arm_compute::ITensor *indices,
                              arm_compute::ITensor *output, int axis)
{
  _input = input;
  _indices = indices;
  _output = output;
  _axis = axis;

  arm_compute::PermutationVector pv;
  if (shouldPermute(input->info(), output->info()))
  {
    // NOTE This vector comes from CLPermuteKernel implementation
    //
    // This implementation permutes a tensor of shape C / W / H into another tensor of shape W / H /
    // C
    //
    //     Original | Permuted
    // 0 | C        | W (from 1)
    // 1 | W        | H (from 2)
    // 2 | H        | C (from 0)
    //
    pv = arm_compute::PermutationVector{1, 2, 0};
  }

  if (utils::isGpuMode())
  {
    if (shouldPermute(input->info(), output->info()))
    {
      _cl_gather.configure(CAST_CL(input), CAST_CL(indices), &_cl_permuted, axis);
      _cl_permute.configure(&_cl_permuted, CAST_CL(output), pv);

      // NOTE _permuted is inaccessible from outside, and thus it is safe to invoke allocate here.
      _cl_permuted.allocator()->allocate();
    }
    else
    {
      _cl_gather.configure(CAST_CL(input), CAST_CL(indices), CAST_CL(output), axis);
    }
  }
  else
  {
    throw std::runtime_error("Not supported, yet");
  }
}

void GenericGather::run(void)
{
  if (utils::isGpuMode())
  {
    _cl_gather.run();
    if (shouldPermute(_input->info(), _output->info()))
    {
      _cl_permute.run();
    }
  }
  else
  {
    throw std::runtime_error("Not supported, yet");
  }
}

} // namespace misc
} // namespace arm_compute
