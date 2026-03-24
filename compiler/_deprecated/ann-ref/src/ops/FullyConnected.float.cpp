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

#include "FullyConnected.float.h"
#include "Assert.h"

#include "internal/Matrix.h"
#include "internal/Fused.h"
#include "internal/GEMM.h"
#include "internal/ActivationUtils.h"

// From optimized_ops.h in TensorFlow Lite
template <FusedActivationFunctionType Ac>
void FullyConnected(const float *input_data, const Dims<4> &input_dims, const float *weights_data,
                    const Dims<4> &weights_dims, const float *bias_data, const Dims<4> &bias_dims,
                    float *output_data, const Dims<4> &output_dims)
{
  // TODO(b/62193649): this convoluted shape computation (determining
  // input_rows from the weights_dims, then MapAsMatrixWithGivenNumberOfRows)
  // is because the current --variable_batch hack consists in overwriting the
  // 3rd dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  // When that is fixed, this should become:
  // const auto input_matrix_map =
  //     MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const int input_rows = ArraySize(weights_dims, 0);
  const auto input_matrix_map =
      MapAsMatrixWithGivenNumberOfRows(input_data, input_dims, input_rows);
  const auto filter_matrix_map = MapAsMatrixWithFirstDimAsRows(weights_data, weights_dims);
  auto output_matrix_map = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);
  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data, output_dims);
}

bool fullyConnectedFloat32(const float *inputData, const Shape &inputShape,
                           const float *weightsData, const Shape &weightsShape,
                           const float *biasData, const Shape &biasShape, int32_t activation,
                           float *outputData, const Shape &outputShape)
{

#define ANDROID_NN_FULLY_CONNECTED(activation)                                                  \
  FullyConnected<FusedActivationFunctionType::activation>(                                      \
      inputData, convertShapeToDims(inputShape), weightsData, convertShapeToDims(weightsShape), \
      biasData, convertShapeToDims(biasShape), outputData, convertShapeToDims(outputShape))

  ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_FULLY_CONNECTED)
#undef ANDROID_NN_FULLY_CONNECTED
  return true;
}
