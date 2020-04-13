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

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/runtime/misc/functions/GenericReshapeLayer.h"

namespace arm_compute
{
namespace misc
{

namespace
{

bool shouldPermute(const arm_compute::ITensorInfo *input, arm_compute::ITensorInfo *output)
{
  return (input->num_dimensions() == 4 || output->num_dimensions() == 4) &&
         (input->num_dimensions() != output->num_dimensions() &&
          input->data_layout() == DataLayout::NCHW);
}

} // namespace

void GenericReshapeLayer::configure(const arm_compute::ITensor *input, arm_compute::ITensor *output)
{
  _input = input;
  _output = output;

  arm_compute::PermutationVector pv;
  if (input->info()->data_layout() == DataLayout::NCHW && input->info()->num_dimensions() == 4 &&
      output->info()->num_dimensions() != 4)
  {
    // NOTE This vector comes from CLPermuteKernel implementation
    //
    // This implementation permutes a tensor of shape W / H / C into another tensor of shape
    // C / W / H
    //
    //     Original | Permuted
    // 0 | W        | C (from 2)
    // 1 | H        | W (from 0)
    // 2 | C        | H (from 1)
    //
    pv = arm_compute::PermutationVector{2, 0, 1};
  }
  else if (input->info()->data_layout() == DataLayout::NCHW &&
           input->info()->num_dimensions() != 4 && output->info()->num_dimensions() == 4)
  {
    // NOTE This vector comes from CLPermuteKernel implementation
    //
    // This implementation permutes a tensor of shape C / W / H into another tensor of shape
    // W / H / C
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
    const auto const_input = CAST_CL(const_cast<arm_compute::ITensor *>(input));
    if (shouldPermute(input->info(), output->info()))
    {
      _cl_permute.configure(const_input, &_cl_permuted, pv);
      _cl_reshape.configure(&_cl_permuted, CAST_CL(output));

      // NOTE _permuted is inaccessible from outside, and thus it is safe to invoke allocate here.
      _cl_permuted.allocator()->allocate();
    }
    else
    {
      _cl_reshape.configure(const_input, CAST_CL(output));
    }
  }
  else
  {
    if (shouldPermute(input->info(), output->info()))
    {
      _neon_permute.configure(input, &_neon_permuted, pv);
      _neon_reshape.configure(&_neon_permuted, output);

      // NOTE _permuted is inaccessible from outside, and thus it is safe to invoke allocate here.
      _neon_permuted.allocator()->allocate();
    }
    else
    {
      _neon_reshape.configure(input, output);
    }
  }
}

void GenericReshapeLayer::run(void)
{
  if (utils::isGpuMode())
  {
    if (shouldPermute(_input->info(), _output->info()))
    {
      _cl_permute.run();
    }
    _cl_reshape.run();
  }
  else
  {
    if (shouldPermute(_input->info(), _output->info()))
    {
      _neon_permute.run();
    }
    _neon_reshape.run();
  }
}

} // namespace misc
} // namespace arm_compute
