/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_PADDIGN_H__
#define __ONERT_IR_PADDIGN_H__

#include "Shape.h"
#include "InternalType.h"

#include <cstdint>
#include <string>

namespace onert
{
namespace ir
{

enum class PaddingType
{
  EXPLICIT = 0,
  SAME = 1,
  VALID = 2
};

/**
 * @brief     Converts a internal padding type to const char*
 * @param[in] type  Padding type to be converted
 * @return    A string holding the converted value
 */
inline std::string to_string(const PaddingType type);

struct ExplicitPadding
{
  uint32_t left;
  uint32_t right;
  uint32_t top;
  uint32_t bottom;
};

// TODO Resolve explicit padding param at frontend and save in value field
struct Padding
{
  Padding(void);
  Padding(PaddingType paddingType);
  Padding(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom);

  // TODO Change to private field
  PaddingType type;
  ExplicitPadding param;
};

// TODO Change to Padding struct's method
const ExplicitPadding calculatePadding(const Padding &padding, const FeatureShape &ifm_shape,
                                       const FeatureShape &ofm_shape, const Stride &stride,
                                       uint32_t kw, uint32_t kh, uint32_t dwf = 1,
                                       uint32_t dhf = 1);

} // namespace ir
} // namespace onert

#endif
