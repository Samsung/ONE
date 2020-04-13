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

#include "Softmax.float.h"
#include "Logging.h"

#include "internal/Array.h"
#include "internal/Matrix.h"

// From optimized_ops.h in TensorFlow Lite
inline void Softmax(const float *input_data, const Dims<4> &input_dims, float beta,
                    float *output_data, const Dims<4> &output_dims)
{
  MatchingArraySize(input_dims, 3, output_dims, 3);
  MatchingArraySize(input_dims, 2, output_dims, 2);
  MatchingArraySize(input_dims, 1, output_dims, 1);
  MatchingArraySize(input_dims, 0, output_dims, 0);

  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
  // Compute the exponential first, removing the max coefficient for numerical
  // stability.
  out_mat = (in_mat.rowwise() - in_mat.colwise().maxCoeff()).array() * beta;
  // We are separating out the exp function so that exp can be vectorized.
  out_mat = out_mat.array().exp();
  // Normalize to get the activations.
  Eigen::Array<float, 1, Eigen::Dynamic> scale = out_mat.array().colwise().sum().inverse();
  out_mat.array().rowwise() *= scale;
}

bool softmaxFloat32(const float *inputData, const Shape &inputShape, const float beta,
                    float *outputData, const Shape &outputShape)
{
  Dims<4> dim;
  if (getNumberOfDimensions(inputShape) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(inputShape, 0);
    uint32_t input_size = getNumberOfElements(inputShape) / batch_size;

    Shape shapeIn4D;
    shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
    dim = convertShapeToDims(shapeIn4D);
  }
  else if (getNumberOfDimensions(inputShape) == 4)
  {
    dim = convertShapeToDims(inputShape);
  }
  else
  {
    LOG(ERROR) << "only 2D and 4D tensors supported";
    return false;
  }

  Softmax(inputData, dim, beta, outputData, dim);
  return true;
}
