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
 * @file        GenericReshapeLayer.h
 * @brief       This file contains GenericReshapeLayer class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_MISC_GENERIC_RESHAPE_LAYER_H__
#define __ARM_COMPUTE_MISC_GENERIC_RESHAPE_LAYER_H__

#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>

#include <arm_compute/runtime/CL/functions/CLPermute.h>
#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPermute.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

#include "Utils.h"

namespace arm_compute
{
namespace misc
{

/**
 * @brief Class to run Reshape Layer with both CPU and GPU
 */
class GenericReshapeLayer : public arm_compute::IFunction
{
public:
  GenericReshapeLayer(void)
      : _input(nullptr), _output(nullptr), _cl_permuted{}, _neon_permuted{}, _cl_permute{},
        _cl_reshape{}, _neon_permute{}, _neon_reshape{}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Configure the layer
   * @param[in] input The source tensor
   * @param[in] output The destination tensor
   * @return N/A
   */
  void configure(const arm_compute::ITensor *input, arm_compute::ITensor *output);

public:
  /**
   * @brief Run the operation. Must be called after configure().
   * @return N/A
   */
  void run(void) override;

private:
  const arm_compute::ITensor *_input;
  arm_compute::ITensor *_output;
  arm_compute::CLTensor _cl_permuted;
  arm_compute::Tensor _neon_permuted;

private:
  arm_compute::CLPermute _cl_permute;
  arm_compute::CLReshapeLayer _cl_reshape;

  arm_compute::NEPermute _neon_permute;
  arm_compute::NEReshapeLayer _neon_reshape;
};

} // namespace misc
} // namespace arm_compute

#endif // __ARM_COMPUTE_MISC_GENERIC_RESHAPE_LAYER_H__
