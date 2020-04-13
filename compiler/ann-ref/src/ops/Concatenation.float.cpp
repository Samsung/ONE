/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "Concatenation.float.h"

#include "internal/Array.h"
#include "internal/Matrix.h"
#include "internal/Fused.h"

// From optimized_ops.h in TensorFlow Lite
template <FusedActivationFunctionType Ac, typename Scalar>
void Concatenation(int concat_dim, const Scalar *const *input_data,
                   const Dims<4> *const *input_dims, int inputs_count, Scalar *output_data,
                   const Dims<4> &output_dims)
{
  DCHECK_GT(inputs_count, 1);
  int concat_size = 0;
  for (int i = 0; i < inputs_count; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      if (j != concat_dim)
      {
        MatchingArraySize(*input_dims[i], j, output_dims, j);
      }
    }
    concat_size += ArraySize(*input_dims[i], concat_dim);
  }
  DCHECK_EQ(concat_size, ArraySize(output_dims, concat_dim));
  DCHECK(IsPackedWithoutStrides(output_dims));
  // for now we dont have a model with a Concatenation
  // with fused activation function.
  DCHECK(Ac == FusedActivationFunctionType::kNone);
  int outer_size = 1;
  for (int i = concat_dim + 1; i < 4; i++)
  {
    outer_size *= output_dims.sizes[i];
  }
  Scalar *output_ptr = output_data;
  for (int k = 0; k < outer_size; k++)
  {
    for (int i = 0; i < inputs_count; ++i)
    {
      const int copy_size = input_dims[i]->sizes[concat_dim] * input_dims[i]->strides[concat_dim];
      memcpy(output_ptr, input_data[i] + k * copy_size, copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

bool concatenationFloat32(const std::vector<const float *> &inputDataPtrs,
                          const std::vector<Shape> &inputShapes, int32_t axis, float *outputData,
                          const Shape &outputShape)
{
  int num_inputs = inputShapes.size();
  std::vector<Dims<4> *> inputDimsPtr(num_inputs);
  std::vector<Dims<4>> inputDims(num_inputs);
  for (int i = 0; i < num_inputs; i++)
  {
    inputDims[i] = convertShapeToDims(inputShapes[i]);
    inputDimsPtr[i] = &inputDims[i];
  }

  Concatenation<FusedActivationFunctionType::kNone, float>(
      getNumberOfDimensions(outputShape) - axis - 1, inputDataPtrs.data(), inputDimsPtr.data(),
      num_inputs, outputData, convertShapeToDims(outputShape));

  return true;
}
