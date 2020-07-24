/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __ARM_COMPUTE_NEFULLYCONNECTEDHYBRIDLAYER_H__
#define __ARM_COMPUTE_NEFULLYCONNECTEDHYBRIDLAYER_H__

#include "arm_compute/core/NEON/kernels/NEQuantizationSymmetricKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/NEON/kernels/NEMuliplyScaleFactorKernel.h"
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCoreEx.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with NEON. This function calls
 * the following kernels:
 *
 *  -# @ref NETransposeKernel
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedHybridLayerReshapeWeights : public INESimpleFunctionNoBorder
{
public:
  /** Set the input and output tensors.
   *
   * @param[in]  input  Weights tensor. The weights must be 2 dimensional. Data types supported:
   * QASYMM8/F16/F32.
   * @param[out] output Destination tensor. Data type supported: Same as @p input.
   */
  void configure(const ITensor *input, ITensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEFullyConnectedHybridLayerReshapeWeights
   *
   * @param[in] input  Weights tensor info. The weights must be 2 dimensional. Data types supported:
   * QASYMM8/F16/F32.
   * @param[in] output Destination tensor info. Data type supported: Same as @p input.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to compute a Fully Connected layer on NEON. This function calls the following
 * NEON kernels:
 *  -# @ref NEIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref NEFullyConnectedHybridLayerReshapeWeights (if @p are_weights_reshaped is set to false
 * and transpose_weights is set to true ) (called once)
 *  -# @ref NEGEMMMatrixMultiplyKernel or @ref NEGEMMLowpMatrixMultiplyCore (if quantized
 * asymmetric)
 *  -# @ref NEGEMMMatrixAccumulateBiasesKernel or @ref
 * NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint (if quantized asymmetric) (if @p biases is
 * not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedHybridLayer : public IFunction
{
public:
  /** Constructor */
  NEFullyConnectedHybridLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NEFullyConnectedHybridLayer(const NEFullyConnectedHybridLayer &) = delete;
  /** Default move constructor */
  NEFullyConnectedHybridLayer(NEFullyConnectedHybridLayer &&) = default;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NEFullyConnectedHybridLayer &operator=(const NEFullyConnectedHybridLayer &) = delete;
  /** Default move assignment operator */
  NEFullyConnectedHybridLayer &operator=(NEFullyConnectedHybridLayer &&) = default;
  /** Set the input and output tensors.
   *
   * @param[in]  input   Source tensor. Data type supported: F16/F32.
   * @param[in]  weights Weights tensor. The weights must be 2 dimensional.
   *                     If this function is called after a Convolution Layer, the (transposed)
   * weights will have as many rows as the product of the first 3 input's dimensions.
   *                     If it is called after another FullyConnected Layer, the (transposed)
   * weights will have as many rows as the input's first dimension.
   *                     Data type supported: S8.
   * @param[in]  biases  Bias tensor. Can be nullptr. Data type supported:Same as @p input.
   * @param[out] output  Destination tensor. Its shape should be equal to the output of a matrix
   * multiplication between:
   *                     - The output of im2col on the input and the (transposed) 2D weights, if the
   * function is called after a Convolution Layer
   *                     - The input tensor and the (transposed) 2D weights, if the function is
   * called after another FullyConnected Layer.
   *                     Data type supported: Same as @p input.
   * @param[in]  fc_info (Optional) Fully connected layer additional info
   */
  void configure(const ITensor *input, const ITensor *weights, const ITensor *biases,
                 ITensor *output, FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEFullyConnectedHybridLayer
   *
   * @param[in]  input   Source tensor info. Data type supported: F16/F32.
   * @param[in]  weights Weights tensor info. The weights must be 2 dimensional.
   *                     If this function is called after a Convolution Layer, the (transposed)
   * weights will have as many rows as the product of the first 3 input's dimensions.
   *                     If it is called after another FullyConnected Layer, the (transposed)
   * weights will have as many rows as the input's first dimension.
   *                     Data type supported: S8.
   * @param[in]  biases  Bias tensor info. Can be nullptr. Data type supported:Same as @p input.
   * @param[out] output  Destination tensor info. Its shape should be equal to the output of a
   * matrix multiplication between:
   *                     - The output of im2col on the input and the (transposed) 2D weights, if the
   * function is called after a Convolution Layer
   *                     - The input tensor and the (transposed) 2D weights, if the function is
   * called after another FullyConnected Layer.
   *                     Data type supported: Same as @p input.
   * @param[in]  fc_info (Optional) Fully connected layer additional info
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *weights,
                         const ITensorInfo *biases, const ITensorInfo *output,
                         FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

  // Inherited methods override
  void run() override;
  void prepare() override;

private:
  void configure_mm(const ITensor *input, const ITensor *weights, ITensor *output);

  MemoryGroup _memory_group;
  NEFullyConnectedHybridLayerReshapeWeights _reshape_weights_function;
  NEQuantizationSymmetricKernel _quant_input_kernel;
  NEGEMMLowpMatrixMultiplyCoreEx _mm_gemmlowp;
  NEMultiplyScaleFactorKernel _multiply_scale_kernel;
  NEGEMMMatrixAccumulateBiasesKernel _accumulate_biases_kernel;
  Tensor _reshape_weights_output;
  Tensor _quantized_input;
  Tensor _scale_factor;
  Tensor _gemmlowp_output;
  const ITensor *_original_weights;
  bool _are_weights_reshaped;
  bool _accumulate_biases;
  bool _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEFULLYCONNECTEDHYBRIDLAYER_H__ */
