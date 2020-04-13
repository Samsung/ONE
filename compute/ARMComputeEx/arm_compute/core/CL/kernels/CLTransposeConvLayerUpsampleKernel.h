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
 * Copyright (c) 2016-2018 ARM Limited.
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

#ifndef __ARM_COMPUTE_CLTRANSPOSECONVLAYERUPSAMPLEKERNEL_H__
#define __ARM_COMPUTE_CLTRANSPOSECONVLAYERUPSAMPLEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the Upsampling layer kernel for transpose convolution on OpenCL.
 */
class CLTransposeConvLayerUpsampleKernel : public ICLKernel
{
public:
  /** Constructor */
  CLTransposeConvLayerUpsampleKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTransposeConvLayerUpsampleKernel(const CLTransposeConvLayerUpsampleKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLTransposeConvLayerUpsampleKernel &
  operator=(const CLTransposeConvLayerUpsampleKernel &) = delete;
  /** Default Move Constructor. */
  CLTransposeConvLayerUpsampleKernel(CLTransposeConvLayerUpsampleKernel &&) = default;
  /** Default move assignment operator */
  CLTransposeConvLayerUpsampleKernel &operator=(CLTransposeConvLayerUpsampleKernel &&) = default;
  /** Default destructor */
  ~CLTransposeConvLayerUpsampleKernel() = default;

  /** Initialise the kernel's input and output.
   *
   * @param[in]  input        Source tensor. Data types supported: QASYMM8/F16/F32.
   * @param[out] output       Destination tensor. Data types supported: same as @p input. All but
   * the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only
   * performed within the XY-plane.
   * @param[in]  inner_border Top and right inner border sizes. These rows and columns will be
   * filled with zero.
   * @param[in]  info         Contains padding and stride information described in @ref
   * PadStrideInfo.
   */
  void configure(const ICLTensor *input, ICLTensor *output, const BorderSize &inner_border,
                 const PadStrideInfo &info);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLTransposeConvLayerUpsample
   *
   * @param[in] input        Source tensor info. Data types supported: QASYMM8/F16/F32.
   * @param[in] output       Destination tensor info. Data types supported: same as @p input. All
   * but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is
   * only performed within the XY-plane.
   * @param[in] inner_border Top and right inner border sizes. These rows and columns will be filled
   * with zero.
   * @param[in] info         Contains padding and stride information described in @ref
   * PadStrideInfo.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const BorderSize &inner_border, const PadStrideInfo &info);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;
  ICLTensor *_output;
  BorderSize _inner_border;
  PadStrideInfo _info;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLTRANSPOSECONVLAYERUPSAMPLEKERNEL_H__ */
