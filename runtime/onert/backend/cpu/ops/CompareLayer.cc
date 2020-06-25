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
#include "CompareLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Comparison.h>

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

using OpType = onert::ir::operation::Comparison::ComparisonType;
using namespace onert::backend::cpu;

template <typename T>
void compareScalar(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
                   OpType op_type)
{
  bool requires_broadcast = !HaveSameShapes(lhs, rhs);

  if (requires_broadcast)
  {
    switch (op_type)
    {
      case OpType::Equal:
        Broadcast4DSlowEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        Broadcast4DSlowNotEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        Broadcast4DSlowGreater(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        Broadcast4DSlowGreaterEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        Broadcast4DSlowLess(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                            getExtendedTensorShape(output),
                            reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        Broadcast4DSlowLessEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  else // if (requires_broadcast == false)
  {
    switch (op_type)
    {
      case OpType::Equal:
        EqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                       getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                       getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        NotEqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                          getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                          getExtendedTensorShape(output),
                          reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        GreaterNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                         getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                         getExtendedTensorShape(output),
                         reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        GreaterEqualNoScaling(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        LessNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                      getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                      getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        LessEqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                           getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                           getExtendedTensorShape(output),
                           reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  return;
}
} // namespace

CompareLayer::CompareLayer()
    : _lhs(nullptr), _rhs(nullptr), _output(nullptr),
      _op_type(ir::operation::Comparison::ComparisonType::Equal)
{
  // DO NOTHING
}

void CompareLayer::compareQuant8() { throw std::runtime_error{"Compare NYI for quantized"}; }

void CompareLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                             const OpType op_type, IPortableTensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _op_type = op_type;
  _output = output;
}

void CompareLayer::run()
{
  if (_lhs->data_type() == OperandType::FLOAT32)
  {
    compareScalar<float>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::INT32)
  {
    compareScalar<int32_t>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::BOOL8)
  {
    compareScalar<uint8_t>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    compareQuant8();
  }
  else
  {
    throw std::runtime_error{"Compare: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
