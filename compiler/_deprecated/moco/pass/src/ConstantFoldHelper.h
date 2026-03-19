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

#ifndef __MOCO_CONSTANT_FOLD_HELPER_H__
#define __MOCO_CONSTANT_FOLD_HELPER_H__

#include <moco/IR/Nodes/TFConst.h>

#include <loco.h>
#include <loco/IR/TensorShape.h>

namespace moco
{

TFConst *new_const(loco::Graph *graph, loco::TensorShape &tensor_shape,
                   const loco::DataType &dtype);

template <typename T> T scalar_from_const(const TFConst *tfconst);
template <> int32_t scalar_from_const<int32_t>(const TFConst *tfconst);
template <> float scalar_from_const<float>(const TFConst *tfconst);

/**
 * @note Check if it is valid to run Constant folding for binary operations
 *       as-of current implementation. That is currently we support for
 *       element-wise or one of the input is scalar.
 *       TODO Support other shapes of binary operation
 */
bool valid_shape_for_constfold_binary_op(const loco::TensorShape &lhs,
                                         const loco::TensorShape &rhs);

struct BinaryFunc
{
  virtual ~BinaryFunc() = default;

  virtual float apply(float, float) const;
  virtual int32_t apply(int32_t, int32_t) const;
};

template <typename T>
void apply_binary(const moco::TFConst *x_const, const moco::TFConst *y_const,
                  moco::TFConst *output_const, const moco::BinaryFunc &f);
template <>
void apply_binary<int32_t>(const moco::TFConst *x_const, const moco::TFConst *y_const,
                           moco::TFConst *output_const, const moco::BinaryFunc &f);
template <>
void apply_binary<float>(const moco::TFConst *x_const, const moco::TFConst *y_const,
                         moco::TFConst *output_const, const moco::BinaryFunc &f);

} // namespace moco

#endif // __MOCO_CONSTANT_FOLD_HELPER_H__
