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

#include "ElementwiseUnaryLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Dequantize.h>
#include <cker/operation/Elementwise.h>
#include <cker/operation/Erf.h>
#include <cker/operation/Exp.h>
#include <cker/operation/LogicalNot.h>
#include <cker/operation/Round.h>

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
void absFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Abs(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

template <typename FromT>
void castPtr(const FromT *in, DataPtr out, int num_elements, ir::DataType data_type_out)
{
  switch (data_type_out)
  {
    case ir::DataType::FLOAT32:
      std::transform(in, in + num_elements, out.f, [](FromT a) { return static_cast<float>(a); });
      return;
    case ir::DataType::INT32:
      std::transform(in, in + num_elements, out.i32,
                     [](FromT a) { return static_cast<int32_t>(a); });
      return;
    case ir::DataType::UINT32:
      std::transform(in, in + num_elements, out.u32,
                     [](FromT a) { return static_cast<uint32_t>(a); });
      return;
    case ir::DataType::UINT8:
      std::transform(in, in + num_elements, out.u8,
                     [](FromT a) { return static_cast<uint8_t>(a); });
      return;
    case ir::DataType::BOOL8:
      static_assert(sizeof(bool) == 1, "cpu backend supports bool type which is 1 byte");
      std::transform(in, in + num_elements, out.b, [](FromT a) { return static_cast<bool>(a); });
      return;
    case ir::DataType::INT64:
      std::transform(in, in + num_elements, out.i64,
                     [](FromT a) { return static_cast<int64_t>(a); });
      return;
    default:
      throw std::runtime_error("Cast: Not supported output type" +
                               std::to_string((int)data_type_out));
  }
}

void cast(const IPortableTensor *input, IPortableTensor *output)
{
  auto input_buf = input->buffer();
  auto output_buf = output->buffer();
  const auto in = *reinterpret_cast<const DataPtr *>(&input_buf);
  auto out = *reinterpret_cast<DataPtr *>(&output_buf);

  auto input_shape = getShape(input);
  auto output_shape = getShape(output);
  const auto num_elements = MatchingFlatSize(input_shape, output_shape);

  switch (input->data_type())
  {
    case ir::DataType::FLOAT32:
      castPtr(in.f, out, num_elements, output->data_type());
      return;
    case ir::DataType::INT32:
      castPtr(in.i32, out, num_elements, output->data_type());
      return;
    case ir::DataType::UINT32:
      castPtr(in.u32, out, num_elements, output->data_type());
      return;
    case ir::DataType::UINT8:
      castPtr(in.u8, out, num_elements, output->data_type());
      return;
    case ir::DataType::BOOL8:
      castPtr(in.b, out, num_elements, output->data_type());
      return;
    case ir::DataType::INT64:
      castPtr(in.i64, out, num_elements, output->data_type());
      return;
    default:
      throw std::runtime_error("Cast: unsupported data type" +
                               std::to_string((int)input->data_type()));
  }
}

void cosFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Cos(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

void dequantizeInt8(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Dequantize(getShape(input), getBuffer<int8_t>(input), getShape(output),
                         getBuffer<float>(output), input->data_scale(), input->data_zero_point());
}

void dequantizeUint8(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Dequantize(getShape(input), getBuffer<uint8_t>(input), getShape(output),
                         getBuffer<float>(output), input->data_scale(), input->data_zero_point());
}

void expFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Exp(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

void erfFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Erf(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

void floorFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Floor(getShape(input), getBuffer<float>(input), getShape(output),
                    getBuffer<float>(output));
}

void logFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Log(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

void logicalNot(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::LogicalNot(getShape(input), getBuffer<bool>(input), getShape(output),
                         getBuffer<bool>(output));
}

template <typename T> void neg(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Neg<T>(getShape(input), getBuffer<T>(input), getShape(output), getBuffer<T>(output));
}

void roundFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Round(getShape(input), getBuffer<float>(input), getShape(output),
                    getBuffer<float>(output));
}

void rsqrtFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Rsqrt(getShape(input), getBuffer<float>(input), getShape(output),
                    getBuffer<float>(output));
}

void sinFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Sin(getShape(input), getBuffer<float>(input), getShape(output),
                  getBuffer<float>(output));
}

void sqrtFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Sqrt(getShape(input), getBuffer<float>(input), getShape(output),
                   getBuffer<float>(output));
}

void squareFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  nnfw::cker::Square(getShape(input), getBuffer<float>(input), getShape(output),
                     getBuffer<float>(output));
}

template <typename T> void zerosLikeFloat32(const IPortableTensor *input, IPortableTensor *output)
{
  if (!HaveSameShapes(input, output))
    throw std::runtime_error{"ZerosLike: input and output shape don't match."};

  auto element_size = getShape(input).FlatSize();

  memset(getBuffer<T>(output), 0, element_size * sizeof(T));
}
} // namespace

void ElementwiseUnaryLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                      const ElementwiseUnaryType op_type)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _output = output;

  switch (op_type)
  {
    case ElementwiseUnaryType::kAbs:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = absFloat32;
      }
      else
      {
        throw std::runtime_error{"Abs: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kCast:
      _kernel = cast;
      break;
    case ElementwiseUnaryType::kCos:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = cosFloat32;
      }
      else
      {
        throw std::runtime_error{"Cos: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kDequantize:
      if ((input->data_type() == OperandType::QUANT_UINT8_ASYMM))
      {
        _kernel = dequantizeUint8;
      }
      else if ((input->data_type() == OperandType::QUANT_INT8_ASYMM) ||
               (input->data_type() == OperandType::QUANT_INT8_SYMM))
      {
        _kernel = dequantizeInt8;
      }
      else
      {
        throw std::runtime_error{"Dequantize: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kExp:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = expFloat32;
      }
      else
      {
        throw std::runtime_error{"Exp: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kErf:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = erfFloat32;
      }
      else
      {
        throw std::runtime_error{"Exp: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kFloor:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = floorFloat32;
      }
      else
      {
        throw std::runtime_error{"Floor: Unsupported data type"};
      }
      break;
    case ElementwiseUnaryType::kLog:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = logFloat32;
      }
      else
      {
        throw std::runtime_error{"Log: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kLogicalNot:
      if ((input->data_type() == OperandType::BOOL8))
      {
        static_assert(sizeof(bool) == 1, "cpu backend supports bool type which is 1 byte");
        _kernel = logicalNot;
      }
      else
      {
        throw std::runtime_error{"LogicalNot: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kNeg:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = neg<float>;
      }
      else if ((input->data_type() == OperandType::INT64))
      {
        _kernel = neg<int64_t>;
      }
      else if ((input->data_type() == OperandType::INT32))
      {
        _kernel = neg<int32_t>;
      }
      else
      {
        throw std::runtime_error{"Neg: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kRound:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = roundFloat32;
      }
      else
      {
        throw std::runtime_error{"Round: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kRSqrt:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = rsqrtFloat32;
      }
      else
      {
        throw std::runtime_error{"RSqrt: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kSin:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = sinFloat32;
      }
      else
      {
        throw std::runtime_error{"Sin: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kSqrt:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = sqrtFloat32;
      }
      else
      {
        throw std::runtime_error{"Sqrt: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kSquare:
      if ((input->data_type() == OperandType::FLOAT32))
      {
        _kernel = squareFloat32;
      }
      else
      {
        throw std::runtime_error{"Square: Unsupported  data type"};
      }
      break;
    case ElementwiseUnaryType::kZerosLike:
      if (input->data_type() == OperandType::FLOAT32)
      {
        _kernel = zerosLikeFloat32<float>;
      }
      else if (input->data_type() == OperandType::INT32)
      {
        _kernel = zerosLikeFloat32<int32_t>;
      }
      else
      {
        throw std::runtime_error{"ZerosLike: Unsupported data type"};
      }
      break;
    default:
      throw std::runtime_error{"ElementwiseUnary: Unsupported ElementwiseUnary type"};
  }
}

void ElementwiseUnaryLayer::run() { _kernel(_input, _output); }

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
