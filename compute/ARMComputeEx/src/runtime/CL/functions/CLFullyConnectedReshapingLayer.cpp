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

#include "arm_compute/runtime/CL/functions/CLFullyConnectedReshapingLayer.h"

#include <arm_compute/runtime/CL/functions/CLFullyConnectedHybridLayer.h>
#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h>
#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayerEx.h>
#include "src/core/helpers/AutoConfiguration.h"

using namespace arm_compute;

void CLFullyConnectedReshapingLayer::configure(const arm_compute::ICLTensor *input,
                                               const arm_compute::ICLTensor *weights,
                                               const arm_compute::ICLTensor *biases,
                                               arm_compute::ICLTensor *output, bool needs_reshape,
                                               const arm_compute::TensorShape &reshape,
                                               KernelType kernel_type)
{
  _input = input;
  _weights = weights;
  _biases = biases;
  _output = output;
  _needs_reshape = needs_reshape;

  const ICLTensor *input_to_use = input;
  if (_needs_reshape)
  {
    // reshape
    auto_init_if_empty(*_cl_buffer.info(),
                       _input->info()->clone()->set_tensor_shape(reshape).set_data_layout(
                         _input->info()->data_layout()));
    _cl_reshape.configure(_input, &_cl_buffer);
    input_to_use = &_cl_buffer;
  }

  _cl_fc = [&]() {
    if (kernel_type == KernelType::GENERAL)
    {
      auto fc = new arm_compute::CLFullyConnectedLayerEx{_memory_manager};
      fc->configure(input_to_use, _weights, _biases, _output);
      return std::unique_ptr<arm_compute::IFunction>(fc);
    }
    else if (kernel_type == KernelType::PREPROCESSED_WEIGHTS)
    {
      bool is_hybrid = (input->info()->data_type() == DataType::F32 ||
                        input->info()->data_type() == DataType::F16) &&
                       (weights->info()->data_type() == DataType::QSYMM8 ||
                        weights->info()->data_type() == DataType::QASYMM8_SIGNED);

      if (is_hybrid)
      {
        auto fc = new arm_compute::CLFullyConnectedHybridLayer{_memory_manager};
        ITensorInfo *weights_info = const_cast<ITensorInfo *>(_weights->info());
        const auto orgin_weights_data_type = weights_info->data_type();
        weights_info->set_data_type(DataType::QASYMM8_SIGNED);
        fc->configure(input_to_use, _weights, _biases, _output);
        weights_info->set_data_type(orgin_weights_data_type);
        return std::unique_ptr<arm_compute::IFunction>(fc);
      }
      else
      {
        auto fc = new arm_compute::CLFullyConnectedLayer{_memory_manager};
        fc->configure(input_to_use, _weights, _biases, _output);
        return std::unique_ptr<arm_compute::IFunction>(fc);
      }
    }
    else
    {
      throw std::runtime_error("CLFullyConnectedReshapingLayer: Unsupported kernel type");
    }
  }();

  if (_needs_reshape)
  {
    // NOTE _cl_buffer is inaccessible from outside, and thus it is safe to invoke allocate here.
    _cl_buffer.allocator()->allocate();
  }
}

void CLFullyConnectedReshapingLayer::run(void)
{
  if (_needs_reshape)
    _cl_reshape.run();

  _cl_fc->run();
}

void CLFullyConnectedReshapingLayer::prepare(void) { _cl_fc->prepare(); }
