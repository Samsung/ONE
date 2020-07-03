/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_INTERP_OPERATIONS_OPERATION_UTILS_H_
#define __ONERT_INTERP_OPERATIONS_OPERATION_UTILS_H_

#include "ir/Shape.h"
#include "ir/InternalType.h"
#include "ir/Padding.h"

#include <cker/Shape.h>
#include <cker/Types.h>

namespace onert
{
namespace interp
{

inline nnfw::cker::Shape convertShape(const ir::Shape &shape)
{
  auto dimensions = std::vector<uint32_t>(shape.dims().begin(), shape.dims().end());

  std::vector<int32_t> raw_shape;
  raw_shape.resize(dimensions.size());

  for (uint32_t i = 0; i < dimensions.size(); ++i)
  {
    raw_shape[i] = dimensions[i];
  }

  return nnfw::cker::GetShape(raw_shape);
}

inline nnfw::cker::Shape convertExtendShape(const ir::Shape &shape)
{
  auto dimensions = std::vector<uint32_t>(shape.dims().begin(), shape.dims().end());

  const int32_t extended_rank = 4;
  int32_t raw_shape[extended_rank];
  uint32_t start = extended_rank - dimensions.size();

  for (uint32_t i = 0; i < extended_rank; ++i)
  {
    if (i < start)
    {
      raw_shape[i] = 1;
    }
    else
    {
      raw_shape[i] = dimensions[i - start];
    }
  }

  return nnfw::cker::Shape(extended_rank, raw_shape);
}

inline nnfw::cker::FusedActivationFunctionType
convertActivationType(const ir::Activation activation)
{
  switch (activation)
  {
    case ir::Activation::NONE:
      return nnfw::cker::FusedActivationFunctionType::kNone;
    case ir::Activation::RELU:
      return nnfw::cker::FusedActivationFunctionType::kRelu;
    case ir::Activation::RELU1:
      return nnfw::cker::FusedActivationFunctionType::kRelu1;
    case ir::Activation::RELU6:
      return nnfw::cker::FusedActivationFunctionType::kRelu6;
    default:
      throw std::runtime_error{"CPU backend: Cannot convert activation type"};
  }
}

template <typename T>
void calculateActivationRange(ir::Activation activation, T *activation_min, T *activation_max)
{
  if (activation == ir::Activation::RELU)
  {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  }
  else if (activation == ir::Activation::RELU6)
  {
    *activation_min = 0;
    *activation_max = 6;
  }
  else if (activation == ir::Activation::RELU1)
  {
    *activation_min = -1;
    *activation_max = 1;
  }
  else if (activation == ir::Activation::NONE)
  {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
  else
  {
    throw std::runtime_error{"Unsupported activation type"};
  }
}

inline ir::Shape calcBroadcastShape(const ir::Shape &lhs, const ir::Shape &rhs, bool &success)
{
  int lhs_rank = lhs.rank();
  int rhs_rank = rhs.rank();

  int out_rank = (lhs_rank > rhs_rank ? lhs_rank : rhs_rank);
  ir::Shape out_shape(out_rank);

  int lhs_idim = lhs_rank - 1;
  int rhs_idim = rhs_rank - 1;
  success = true;
  for (int out_idim = out_rank - 1; out_idim >= 0; out_idim--)
  {
    if (lhs_idim == -1 && rhs_idim == -1)
    {
      // invalid result
      success = false;
      break;
    }

    if (lhs_idim == -1)
    {
      out_shape.dim(out_idim) = rhs.dim(rhs_idim);
      rhs_idim--;
    }
    else if (rhs_idim == -1)
    {
      out_shape.dim(out_idim) = lhs.dim(lhs_idim);
      lhs_idim--;
    }
    else
    {
      if (lhs.dim(lhs_idim) == rhs.dim(rhs_idim))
      {
        out_shape.dim(out_idim) = lhs.dim(lhs_idim);
        lhs_idim--;
        rhs_idim--;
      }
      else if (lhs.dim(lhs_idim) == 1)
      {
        out_shape.dim(out_idim) = rhs.dim(rhs_idim);
        lhs_idim--;
        rhs_idim--;
      }
      else if (rhs.dim(rhs_idim) == 1)
      {
        out_shape.dim(out_idim) = lhs.dim(lhs_idim);
        lhs_idim--;
        rhs_idim--;
      }
      else
      {
        // invalid result
        success = false;
        break;
      }
    }
  }

  if (lhs_idim != -1 || rhs_idim != -1)
  {
    // invalid result
    success = false;
  }
  return out_shape;
}

inline nnfw::cker::PaddingType convertPaddingType(ir::PaddingType ir_padding_type)
{
  switch (ir_padding_type)
  {
    case ir::PaddingType::EXPLICIT:
      return nnfw::cker::PaddingType::kNone;
    case ir::PaddingType::SAME:
      return nnfw::cker::PaddingType::kSame;
    case ir::PaddingType::VALID:
      return nnfw::cker::PaddingType::kValid;
    default:
      throw std::runtime_error("Wrong padding type.");
      break;
  }
}

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_OPERATIONS_OPERATION_UTILS_H_
