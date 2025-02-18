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

// from tensorflow/core/util/padding.cc

#include "Padding.h"
#include "Errors.h"

#include <iostream>

namespace mlir
{
namespace Circle
{

Status GetPaddingFromString(StringPiece str_value, Padding *value)
{
  if (str_value == "SAME")
  {
    *value = SAME;
  }
  else if (str_value == "VALID")
  {
    *value = VALID;
  }
  else if (str_value == "EXPLICIT")
  {
    *value = EXPLICIT;
  }
  else
  {
    std::cerr << str_value << " is not an allowed padding type" << std::endl;
    return Status(Code::ERROR);
  }
  return OkStatus();
}

} // namespace Circle
} // namespace mlir
