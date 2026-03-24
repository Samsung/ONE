/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ARRAY_H__
#define __ARRAY_H__

#include "Shape.h"
#include "Dims.h"

#include "Macro.h"

// From types.h in TensorFlow Lite
//
// Get common array size, DCHECKing that they all agree.
template <typename ArrayType1, typename ArrayType2>
int MatchingArraySize(const ArrayType1 &array1, int index1, const ArrayType2 &array2, int index2)
{
  DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return ArraySize(array1, index1);
}

// From types.h in TensorFlow Lite
template <typename ArrayType1, typename ArrayType2, typename... Args>
int MatchingArraySize(const ArrayType1 &array1, int index1, const ArrayType2 &array2, int index2,
                      Args... args)
{
  DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return MatchingArraySize(array1, index1, args...);
}

#endif // __ARRAY_H__
