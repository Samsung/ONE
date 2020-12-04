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

/**
 * @file        NEFullyConnectedReshapingLayer.h
 * @brief       This file contains NEFullyConnectedReshapingLayer class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_NE_FULLY_CONNECTED_RESHAPING_LAYER_H__
#define __ARM_COMPUTE_NE_FULLY_CONNECTED_RESHAPING_LAYER_H__

#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>
#include <arm_compute/runtime/IMemoryManager.h>
#include <arm_compute/runtime/Tensor.h>

namespace arm_compute
{
/**
 * @brief Class to run FullyConnected Layer after reshaping input tensor
 */
class NEFullyConnectedReshapingLayer : public arm_compute::IFunction
{
public:
  enum class KernelType
  {
    GENERAL,             //< General FC
    PREPROCESSED_WEIGHTS //< Weights are constants so it can be preprocessed
  };

public:
  NEFullyConnectedReshapingLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr)
    : _memory_manager{memory_manager}, _input(nullptr), _weights(nullptr), _biases(nullptr),
      _output(nullptr), _neon_buffer{}, _neon_fc{nullptr}, _neon_reshape{}, _needs_reshape(false)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Configure the layer
   * @param[in] input The source tensor
   * @param[in] weights The tensor that is filled with weight values
   * @param[in] biases The tensor that is filled with biase values
   * @param[in] output The destination tensor
   * @param[in] needs_reshape Whether it needs to be reshaped or not
   * @param[in] reshape The tensor shape to be reshaped. Only valid when needs_reshape is true.
   * @param[in] kernel_type The kernel type for actual FullyConnected layer
   * @return N/A
   */
  void configure(const arm_compute::ITensor *input, const arm_compute::ITensor *weights,
                 const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                 bool needs_reshape, const arm_compute::TensorShape &reshape,
                 KernelType kernel_type);

public:
  /**
   * @brief Run the operation. Must be called after configure().
   * @return N/A
   */
  void run(void) override;
  /**
   * @brief Prepare the operation
   * @return N/A
   */
  void prepare(void) override;

private:
  std::shared_ptr<IMemoryManager> _memory_manager;
  const arm_compute::ITensor *_input;
  const arm_compute::ITensor *_weights;
  const arm_compute::ITensor *_biases;
  arm_compute::ITensor *_output;

  // buffer for reshaping input tensor
  arm_compute::Tensor _neon_buffer;

private:
  std::unique_ptr<arm_compute::IFunction> _neon_fc;
  NEReshapeLayer _neon_reshape;
  bool _needs_reshape;
};
} // namespace arm_compute

#endif // __ARM_COMPUTE_NE_FULLY_CONNECTED_RESHAPING_LAYER_H__
