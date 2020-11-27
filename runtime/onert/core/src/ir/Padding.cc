/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Padding.h"

#include "util/Utils.h"

#include <stdexcept>
#include <cassert>

namespace onert
{
namespace ir
{
namespace
{

inline ExplicitPadding validPadding(void)
{
  //
  // ANEURALNETWORKS_PADDING_VALID
  //
  // VALID padding. No padding.
  //
  // When the input size is not evenly divisible by the filter size,
  // the input at the end that could not fill the whole filter tile
  // will simply be ignored.
  //
  ExplicitPadding padding;

  padding.top = 0;
  padding.bottom = 0;
  padding.left = 0;
  padding.right = 0;

  return padding;
}

inline ExplicitPadding samePaddingUsingIFM(const FeatureShape &ifm_shape, const Stride &stride,
                                           uint32_t kw, uint32_t kh, uint32_t dwf, uint32_t dhf)
{
  ExplicitPadding padding;

  // ANEURALNETWORKS_PADDING_SAME (from NNAPI spec)
  //
  // SAME padding. Padding on both ends are the "same":
  //
  // padding_to_beginning = total_padding / 2
  // padding_to_end = (total_padding + 1)/2.
  //
  const int32_t effective_filter_h_size = (kh - 1) * dhf + 1;
  const int32_t effective_filter_w_size = (kw - 1) * dwf + 1;

  const int32_t vertical_expected_output = (ifm_shape.H + stride.vertical - 1) / stride.vertical;
  const int32_t horizontal_expected_output =
    (ifm_shape.W + stride.horizontal - 1) / stride.horizontal;

  const int32_t vertical_needed_input =
    (vertical_expected_output - 1) * stride.vertical + effective_filter_h_size;
  const int32_t vertical_total_padding = std::max(0, vertical_needed_input - ifm_shape.H);

  const int32_t horizontal_needed_input =
    (horizontal_expected_output - 1) * stride.horizontal + effective_filter_w_size;
  const int32_t horizontal_total_padding = std::max(0, horizontal_needed_input - ifm_shape.W);

  padding.top = vertical_total_padding / 2;
  padding.bottom = (vertical_total_padding + 1) / 2;
  padding.left = horizontal_total_padding / 2;
  padding.right = (horizontal_total_padding + 1) / 2;

  return padding;
}

inline ExplicitPadding samePadding(const FeatureShape &ifm_shape, const FeatureShape &ofm_shape,
                                   const Stride &stride, uint32_t kw, uint32_t kh, uint32_t dwf,
                                   uint32_t dhf)
{
  const int32_t vertical_expected_output = (ifm_shape.H + stride.vertical - 1) / stride.vertical;
  const int32_t horizontal_expected_output =
    (ifm_shape.W + stride.horizontal - 1) / stride.horizontal;
  assert(vertical_expected_output == ofm_shape.H);
  assert(horizontal_expected_output == ofm_shape.W);

  UNUSED_RELEASE(ofm_shape);
  UNUSED_RELEASE(vertical_expected_output);
  UNUSED_RELEASE(horizontal_expected_output);

  return samePaddingUsingIFM(ifm_shape, stride, kw, kh, dwf, dhf);
}

} // namespace

inline std::string to_string(const PaddingType type)
{
  switch (type)
  {
    case PaddingType::EXPLICIT:
      return "Padding::EXPLICIT";
    case PaddingType::SAME:
      return "Padding::SAME";
    case PaddingType::VALID:
      return "Padding::VALID";
    default:
      throw std::runtime_error{"Fail to convert string: wrong padding type"};
  }
}

Padding::Padding(void) : type{PaddingType::EXPLICIT}, param{0, 0, 0, 0}
{
  // DO NOTHING
}

Padding::Padding(PaddingType paddingType) : type{paddingType}, param{0, 0, 0, 0}
{
  assert(paddingType != PaddingType::EXPLICIT);
}

Padding::Padding(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom)
  : type{PaddingType::EXPLICIT}, param{left, right, top, bottom}
{
  // DO NOTHING
}

const ExplicitPadding calculatePadding(const Padding &padding, const FeatureShape &ifm_shape,
                                       const FeatureShape &ofm_shape, const Stride &stride,
                                       uint32_t kw, uint32_t kh, uint32_t dwf, uint32_t dhf)
{
  if (padding.type == PaddingType::EXPLICIT)
  {
    return padding.param;
  }
  else if (padding.type == PaddingType::SAME)
  {
    return samePadding(ifm_shape, ofm_shape, stride, kw, kh, dwf, dhf);
  }
  else if (padding.type == PaddingType::VALID)
  {
    return validPadding();
  }
  else
  {
    throw std::runtime_error{"Cannot handle padding type"};
  }
}

} // namespace ir
} // namespace onert
