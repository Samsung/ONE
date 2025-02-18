/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/core/util/padding.h

#ifndef __CIRCLE_MLIR_DIALECT_UTILS_PADDING_H__
#define __CIRCLE_MLIR_DIALECT_UTILS_PADDING_H__

#include "Errors.h"

#include <absl/strings/string_view.h>

#include <string>

namespace mlir
{
namespace Circle
{

// from using StringPiece = tsl::StringPiece;
// TODO replace with std::string if there is no special usage with absl
using StringPiece = absl::string_view;

// Padding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Three types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
//   EXPLICIT: The user specifies the pad values in the explicit_paddings
//             attribute.
// The padded area is typically zero-filled. For pooling ops, the padded area is
// instead ignored. For max pool, this is equivalent to padding with -infinity.
enum Padding
{
  VALID = 1,    // No padding.
  SAME = 2,     // Input and output layers have the same size.
  EXPLICIT = 3, // Padding is explicitly specified
};

// Sets padding value based on the given string padding value.
Status GetPaddingFromString(StringPiece str_value, Padding *value);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_UTILS_PADDING_H__
