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

/**
 * @file NEHashtableLookup.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::NEHashtableLookup class
 */

#ifndef __ARM_COMPUTE_NEHASHTABLELOOKUP_H__
#define __ARM_COMPUTE_NEHASHTABLELOOKUP_H__

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include <vector>

namespace arm_compute
{
class ITensor;

/**
 * @brief Class to perform HashtableLookup operation
 */
class NEHashtableLookup : public INESimpleFunctionNoBorder
{
public:
  /**
   * @brief Set the input and output tensors.
   * @param[in]  lookups  Lookups 1D tensor that values are indices into the first dimension of
   *                      input. Data types supported: S32
   * @param[in]  keys     Keys 1D tensor. keys and input pair represent a map.
   *                      Data types supported: S32
   * @param[in]  input    Source tensor.
   *                      Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[out] output   Destination tensor. Data types and data layouts supported: Same as @p
   *                      input.
   * @param[out] hits     Hits 1D tensor. A boolean tensor that indicates whether the lookup hits
   *                      (True) or not (False). Data types supported: U8/QASYMM8
   * @return N/A
   */
  void configure(const ITensor *lookups, const ITensor *keys, const ITensor *input, ITensor *output,
                 ITensor *hits);
  /** Static function to check if given info will lead to a valid configuration of @ref NECopy
   *
   * @param[in]  lookups  Lookups 1D tensor info.
   *                      Data types supported: S32
   * @param[in]  keys     Keys 1D tensor info. keys and input pair represent a map.
   *                      Data types supported: S32
   * @param[in]  input    Source tensor info.
   *                      Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[in]  output   Destination tensor info. Data types and data layouts supported: Same as @p
   * input.
   * @param[in]  hits     Hits 1D tensor info. A boolean tensor that indicates whether the lookup
   * hits (True) or not (False). Data types supported: U8/QASYMM8
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *lookups, const ITensorInfo *keys,
                         const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *hits);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEHASHTABLELOOKUP_H__ */
