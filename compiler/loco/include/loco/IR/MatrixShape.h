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

#ifndef __LOCO_IR_MATRIX_SHAPE_H__
#define __LOCO_IR_MATRIX_SHAPE_H__

#include "loco/IR/Dimension.h"

namespace loco
{

/**
 * @brief Matrix Shape
 *
 * This class describes the shape of matrix, which serves as the input/output of
 * matrix operations (e.g. Matrix Multiplication).
 *
 * Each matrix is a collection of 2D features conceptually.
 * Each matrix has height, width.
 *
 * height() refers to the height of matrix in a given matrix
 * width() refers to the width of matrix in a given matrix
 */
class MatrixShape final
{
public:
  MatrixShape() = default;

public:
  const Dimension &height(void) const { return _height; }
  Dimension &height(void) { return _height; }

  const Dimension &width(void) const { return _width; }
  Dimension &width(void) { return _width; }

private:
  Dimension _height;
  Dimension _width;
};

} // namespace loco

#endif // __LOCO_IR_MATRIX_SHAPE_H__
