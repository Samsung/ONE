/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __SPATIAL_H__
#define __SPATIAL_H__

#include <cstdint>

inline uint32_t computeOutSize(uint32_t imageSize, uint32_t filterSize, uint32_t stride,
                               uint32_t paddingHead, uint32_t paddingTail)
{
  return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

#endif // __SPATIAL_H__
