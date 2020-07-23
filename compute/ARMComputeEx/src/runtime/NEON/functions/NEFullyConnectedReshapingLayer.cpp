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

#include "arm_compute/runtime/NEON/functions/NEFullyConnectedReshapingLayer.h"

#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedHybridLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayerEx.h>

using namespace arm_compute;

void NEFullyConnectedReshapingLayer::configure(const arm_compute::ITensor *input,
                                               const arm_compute::ITensor *weights,
                                               const arm_compute::ITensor *biases,
                                               arm_compute::ITensor *output, bool needs_reshape,
                                               const arm_compute::TensorShape &reshape,
                                               KernelType kernel_type)
{
  _input = input;
  _weights = weights;
  _biases = biases;
  _output = output;
  _needs_reshape = needs_reshape;

  const ITensor *input_to_use = input;
  if (_needs_reshape)
  {
    // reshape
    auto_init_if_empty(*_neon_buffer.info(), _input->info()->clone()->set_tensor_shape(reshape));
    _neon_reshape.configure(_input, &_neon_buffer);
    input_to_use = &_neon_buffer;
  }

  _neon_fc = [&]() {
    if (kernel_type == KernelType::GENERAL)
    {
      auto fc = new arm_compute::NEFullyConnectedLayerEx{_memory_manager};
      fc->configure(input_to_use, _weights, _biases, _output);
      return std::unique_ptr<arm_compute::IFunction>(fc);
    }
    else
    {
      assert(kernel_type == KernelType::PREPROCESSED_WEIGHTS);

      bool is_hybrid = input->info()->data_type() == DataType::F32 &&
                       (weights->info()->data_type() == DataType::S8 ||
                        weights->info()->data_type() == DataType::QASYMM8_SIGNED);

      if (is_hybrid)
      {
        auto fc = new arm_compute::NEFullyConnectedHybridLayer{_memory_manager};
        ITensorInfo *weights_info = const_cast<ITensorInfo *>(_weights->info());
        const auto orgin_weights_data_type = weights_info->data_type();
        weights_info->set_data_type(DataType::QASYMM8_SIGNED);
        fc->configure(input_to_use, _weights, _biases, _output);
        weights_info->set_data_type(orgin_weights_data_type);
        return std::unique_ptr<arm_compute::IFunction>(fc);
      }
      else
      {
        auto fc = new arm_compute::NEFullyConnectedLayer{_memory_manager};
        fc->configure(input_to_use, _weights, _biases, _output);
        return std::unique_ptr<arm_compute::IFunction>(fc);
      }
    }
  }();

  // NOTE _neon_buffer is inaccessible from outside, and thus it is safe to invoke allocate here.
  if (_needs_reshape)
  {
    _neon_buffer.allocator()->allocate();
  }
}

void NEFullyConnectedReshapingLayer::run(void)
{
  if (_needs_reshape)
    _neon_reshape.run();

  _neon_fc->run();
}

void NEFullyConnectedReshapingLayer::prepare(void) { _neon_fc->prepare(); }
