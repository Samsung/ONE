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

#ifndef __LOCO_IR_MATRIX_CODEC_H__
#define __LOCO_IR_MATRIX_CODEC_H__

#include "loco/IR/MatrixShape.h"
#include "loco/IR/MatrixIndex.h"

#include "loco/IR/TensorShape.h"
#include "loco/IR/TensorIndex.h"

#include <memory>

namespace loco
{

/**
 * @brief Decribe how to build a matrix from a tensor
 *
 * Let us assume that "enc" is a matrix encoder.
 *
 * Given a tensor "inp" and its shape "inp.shape", "enc" builds a matrix
 * "out" as follows:
 *
 * for each valid matrix index (referred to as matrix_idx below) for enc.shape(inp.shape)
 *   out.at(matrix_index) = inp.at(enc.value(matrix_index))
 */
struct MatrixEncoder
{
  virtual ~MatrixEncoder() = default;

  virtual MatrixShape shape(const TensorShape &shape) const = 0;
  virtual TensorIndex value(const MatrixIndex &index) const = 0;
};

/**
 * @brief Describe how to build a tensor from a matrix
 *
 * Let us assume that "dec" is a matrix decoder.
 *
 * Given a matrix "inp" and its shape "inp.shape", "dec" builds a tensor
 * "out" as follows:
 *
 * for each valid tensor index (referred to as tensor_index below) for dec.shape(inp.shape)
 *   out.at(tensor_index) = inp.at(dec.value(tensor_index))
 *
 * NOTE "inp" is a matrix value and "out" is a tensor value in this example.
 */
struct MatrixDecoder
{
  virtual ~MatrixDecoder() = default;

  virtual TensorShape shape(const MatrixShape &) const = 0;
  virtual MatrixIndex value(const TensorIndex &) const = 0;
};

} // namespace loco

#endif // __LOCO_IR_MATRIX_CODEC_H__
