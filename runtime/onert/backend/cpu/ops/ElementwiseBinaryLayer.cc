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

#include "ElementwiseBinaryLayer.h"

#include "OperationUtils.h"

#include <cker/operation/FloorDiv.h>
#include <cker/operation/FloorMod.h>
#include <cker/operation/LogicalAnd.h>
#include <cker/operation/LogicalOr.h>
#include <cker/operation/MaxMin.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

namespace
{
template <typename T>
void FloorDivGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs,
                     IPortableTensor *output)
{
  if (!HaveSameShapes(lhs, rhs))
  {
    nnfw::cker::FloorDivBroadcast<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs),
                                     getBuffer<T>(rhs), getShape(output), getBuffer<T>(output));
  }
  else
  {
    nnfw::cker::FloorDivElementwise<T>(getShape(lhs), getBuffer<T>(lhs), getBuffer<T>(rhs),
                                       getBuffer<T>(output));
  }
}

template <typename T>
void FloorModGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs,
                     IPortableTensor *output)
{
  if (!HaveSameShapes(lhs, rhs))
  {
    nnfw::cker::FloorModBroadcast<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs),
                                     getBuffer<T>(rhs), getShape(output), getBuffer<T>(output));
  }
  else
  {
    nnfw::cker::FloorModElementwise<T>(getShape(lhs), getBuffer<T>(lhs), getBuffer<T>(rhs),
                                       getBuffer<T>(output));
  }
}

template <typename T>
void logicalAndGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs,
                       IPortableTensor *output)
{
  if (!HaveSameShapes(lhs, rhs))
  {
    nnfw::cker::LogicalAndBroadcast<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs),
                                       getBuffer<T>(rhs), getShape(output), getBuffer<T>(output));
  }
  else
  {
    nnfw::cker::LogicalAndElementwise<T>(getShape(lhs), getBuffer<T>(lhs), getBuffer<T>(rhs),
                                         getBuffer<T>(output));
  }
}

template <typename T>
void logicalOrGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs,
                      IPortableTensor *output)
{
  if (!HaveSameShapes(lhs, rhs))
  {
    nnfw::cker::LogicalOrBroadcast<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs),
                                      getBuffer<T>(rhs), getShape(output), getBuffer<T>(output));
  }
  else
  {
    nnfw::cker::LogicalOrElementwise<T>(getShape(lhs), getBuffer<T>(lhs), getBuffer<T>(rhs),
                                        getBuffer<T>(output));
  }
}

template <typename T>
void maximumGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output)
{
  nnfw::cker::Max<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs), getBuffer<T>(rhs),
                     getShape(output), getBuffer<T>(output));
}

template <typename T>
void minimumGeneric(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output)
{
  nnfw::cker::Min<T>(getShape(lhs), getBuffer<T>(lhs), getShape(rhs), getBuffer<T>(rhs),
                     getShape(output), getBuffer<T>(output));
}

bool haveSameQauntInfo(const IPortableTensor *lhs, const IPortableTensor *rhs,
                       const IPortableTensor *output)
{
  return (lhs->data_scale() == rhs->data_scale() && lhs->data_scale() == output->data_scale()) &&
         (lhs->data_zero_point() == rhs->data_zero_point() &&
          lhs->data_zero_point() == output->data_zero_point());
}
} // namespace

void ElementwiseBinaryLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                                       IPortableTensor *output, const ElementwiseBinaryType op_type)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _output = output;

  switch (op_type)
  {
    case ElementwiseBinaryType::kFloorDiv:
      if (_lhs->data_type() == OperandType::FLOAT32)
      {
        _kernel = FloorDivGeneric<float>;
      }
      else if (_lhs->data_type() == OperandType::INT32)
      {
        _kernel = FloorDivGeneric<int32_t>;
      }
      else
      {
        throw std::runtime_error{"FloorDiv: unsupported data type"};
      }
      break;
    case ElementwiseBinaryType::kFloorMod:
      if (_lhs->data_type() == OperandType::FLOAT32)
      {
        _kernel = FloorDivGeneric<float>;
      }
      else if (_lhs->data_type() == OperandType::INT64)
      {
        _kernel = FloorDivGeneric<int64_t>;
      }
      else
      {
        throw std::runtime_error{"FloorMod: unsupported data type"};
      }
      break;
    case ElementwiseBinaryType::kLogicalAnd:
      if ((_lhs->data_type() == OperandType::BOOL8) && (_rhs->data_type() == OperandType::BOOL8))
      {
        _kernel = logicalAndGeneric<bool>;
      }
      else
      {
        throw std::runtime_error{"LogicalOr: Unsupported data type"};
      }
      break;
    case ElementwiseBinaryType::kLogicalOr:
      if ((_lhs->data_type() == OperandType::BOOL8) && (_rhs->data_type() == OperandType::BOOL8))
      {
        _kernel = logicalOrGeneric<bool>;
      }
      else
      {
        throw std::runtime_error{"LogicalOr: Unsupported data type"};
      }
      break;
    case ElementwiseBinaryType::kMax:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        if (!haveSameQauntInfo(_lhs, _rhs, _output))
        {
          throw std::runtime_error("Max NYI for quantized");
        }
        _kernel = maximumGeneric<uint8_t>;
      }
      else if (_lhs->data_type() == OperandType::FLOAT32)
      {
        _kernel = maximumGeneric<float>;
      }
      else
      {
        throw std::runtime_error{"Max: unsupported data type"};
      }
      break;
    case ElementwiseBinaryType::kMin:
      if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        if (!haveSameQauntInfo(_lhs, _rhs, _output))
        {
          throw std::runtime_error("Min NYI for quantized");
        }
        _kernel = minimumGeneric<uint8_t>;
      }
      else if (_lhs->data_type() == OperandType::INT32)
      {
        _kernel = minimumGeneric<int32_t>;
      }
      else if (_lhs->data_type() == OperandType::FLOAT32)
      {
        _kernel = minimumGeneric<float>;
      }
      else
      {
        throw std::runtime_error{"Min: unsupported data type"};
      }
      break;
    default:
      throw std::runtime_error{"ElementwiseBinary: Unsupported ElementwiseBinary type"};
  }
}

void ElementwiseBinaryLayer::run() { _kernel(_lhs, _rhs, _output); }

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
