/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_BATCHMATMUL_H
#define LUCI_INTERPRETER_PAL_BATCHMATMUL_H

#include <tensorflow/lite/kernels/internal/reference/batch_matmul.h>

namespace luci_interpreter_pal
{
inline void BatchMatMul(const tflite::RuntimeShape &lhs_shape, const float *lhs_data,
                        const tflite::RuntimeShape &rhs_shape, const float *rhs_data,
                        const tflite::RuntimeShape &output_shape, float *output_data)
{
  tflite::reference_ops::BatchMatMul(lhs_shape, lhs_data, rhs_shape, rhs_data, output_shape,
                                     output_data);
}

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *lhs_scratchpad,
                                         luci_interpreter::Tensor *rhs_scratchpad,
                                         const tflite::RuntimeShape &lhs_shape,
                                         const tflite::RuntimeShape &rhs_shape)
{
  // Scratchpad for transposed LHS
  {
    auto lhs_rank = lhs_shape.DimensionsCount();
    luci_interpreter::Shape scratchpad_size(lhs_rank);
    for (int i = 0; i < lhs_rank - 2; ++i)
    {
      scratchpad_size.dim(i) = lhs_shape.Dims(i);
    }
    scratchpad_size.dim(lhs_rank - 2) = lhs_shape.Dims(lhs_rank - 1);
    scratchpad_size.dim(lhs_rank - 1) = lhs_shape.Dims(lhs_rank - 2);

    lhs_scratchpad->resize(scratchpad_size);
  }
  // Scratchpad for transposed RHS
  {
    auto rhs_rank = rhs_shape.DimensionsCount();
    luci_interpreter::Shape scratchpad_size(rhs_rank);
    for (int i = 0; i < rhs_rank - 2; ++i)
    {
      scratchpad_size.dim(i) = rhs_shape.Dims(i);
    }
    scratchpad_size.dim(rhs_rank - 2) = rhs_shape.Dims(rhs_rank - 1);
    scratchpad_size.dim(rhs_rank - 1) = rhs_shape.Dims(rhs_rank - 2);

    rhs_scratchpad->resize(scratchpad_size);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_BATCHMATMUL_H
