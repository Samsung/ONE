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

#include "MatMulConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"
#include "Check.h"

#include <loco.h>
#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

namespace exo
{
/**
 * @brief Converts loco::MatMul to locoex::TFLFullyConnected
 * @note  Because TFLFullyConnected accepts input and weights of loco::Domain::Matrix,
 *        loco::MatrixDecode will be inserted as an input and weights
 *        to meet domain invariant.
 *
 * How it works:
 *
 * Before:
 *   Foo1 ---- MatrixEncode ---- MatMul ---- MatrixDecode ---- Bar
 *   Foo2 ---- MatrixEncode ----/
 *
 * After:
 *
 *   Foo1 - MatrixEncode - MatrixDecode - TFLFullyConnected - MatrixEncode - MatrixDecode - Bar
 *   Foo2 - MatrixEncode - MatrixDecode -/
 *
 * @note  This method replaces MatMul with "- MatrixDecode - TFLFullyConnected - MatrixEncode -".
 *                                          - MatrixDecode -/
 *        Redundant nodes will be removed during transforms.
 *
 * @ref
 * https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/lite/kernels/internal/reference/fully_connected.h
 */
bool MatMulConverter::convert(loco::MatMul *origin)
{
  auto *graph = origin->graph();

  assert(origin->lhs());
  assert(origin->rhs());

  auto tfl_fc = graph->nodes()->create<locoex::TFLFullyConnected>();
  tfl_fc->fusedActivationFunction(locoex::FusedActFunc::NONE);

  // let's create a new graph connection with tfl_fc
  {
    // input
    auto lhs_matrix_dec = make_matrix_decode<MatrixLayout::HW>(origin->lhs());
    tfl_fc->input(lhs_matrix_dec);

    // weights (WH format on TFLite)
    auto rhs_matrix_dec = make_matrix_decode<MatrixLayout::WH>(origin->rhs());
    tfl_fc->weights(rhs_matrix_dec);

    // bias
    auto zero_const = graph->nodes()->create<locoex::TFLConst>();
    { // TODO Create optimization pass which fuse additional Add into bias of Conv or FC
      assert(loco::shape_known(origin));
      assert(loco::dtype_known(origin) && loco::dtype_get(origin) == loco::DataType::FLOAT32);

      auto output_depth = loco::shape_get(origin->rhs()).as<loco::MatrixShape>().width();
      // TODO Fix it with type inference
      zero_const->dtype(loco::DataType::FLOAT32);
      zero_const->rank(1);
      zero_const->dim(0) = output_depth;
      zero_const->size<loco::DataType::FLOAT32>(output_depth.value());
      for (uint32_t x = 0; x < output_depth.value(); x++)
        zero_const->at<loco::DataType::FLOAT32>(x) = 0.0;
    }
    tfl_fc->bias(zero_const);

    // output
    auto matrix_enc = make_matrix_encode<MatrixLayout::HW>(tfl_fc);

    // replace canonical node
    loco::replace(origin).with(matrix_enc);
    origin->lhs(nullptr);
    origin->rhs(nullptr);
  }

  return true;
}

} // namespace exo
