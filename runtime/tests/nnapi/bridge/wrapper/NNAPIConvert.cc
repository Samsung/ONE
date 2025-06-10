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

#include "NNAPIConvert.h"

#include <numeric>

using namespace onert::ir;

DataType NNAPIConvert::getDataType(OperandCode type)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      return DataType::FLOAT32;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      return DataType::INT32;
    case ANEURALNETWORKS_UINT32:
      return DataType::UINT32;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      return DataType::QUANT_UINT8_ASYMM;
    case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
      return DataType::QUANT_INT8_SYMM;
    case ANEURALNETWORKS_BOOL:
    case ANEURALNETWORKS_TENSOR_BOOL8:
      return DataType::BOOL8;
    case ANEURALNETWORKS_TENSOR_FLOAT16:
    case ANEURALNETWORKS_FLOAT16:
      return DataType::FLOAT16;
    case ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL:
      return DataType::QUANT_INT8_SYMM_PER_CHANNEL;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED:
      return DataType::QUANT_INT8_ASYMM;
    default:
      throw std::runtime_error("Unsupported type");
  }
}

TypeInfo NNAPIConvert::getTypeInfo(const ANeuralNetworksOperandType *type)
{
  return TypeInfo(getDataType((OperandCode)(type->type)), type->scale, type->zeroPoint);
}

Shape NNAPIConvert::getShape(const ANeuralNetworksOperandType *type)
{
  Shape shape(type->dimensionCount);

  for (uint32_t axis = 0; axis < type->dimensionCount; ++axis)
  {
    shape.dim(axis) = type->dimensions[axis];
  }

  return shape;
}

size_t NNAPIConvert::calculateSizeFromType(const ANeuralNetworksOperandType *type)
{
  auto shape = getShape(type);
  auto data_type = getDataType((OperandCode)(type->type));

  return shape.num_elements() * sizeOfDataType(data_type);
}

Activation NNAPIConvert::getFusedActivation(FuseCode act)
{
  switch (act)
  {
    case ANEURALNETWORKS_FUSED_NONE:
      return Activation::NONE;
    case ANEURALNETWORKS_FUSED_RELU:
      return Activation::RELU;
    case ANEURALNETWORKS_FUSED_RELU1:
      return Activation::RELU1;
    case ANEURALNETWORKS_FUSED_RELU6:
      return Activation::RELU6;
    default:
      throw std::runtime_error("Unsupported activation type");
  }
}

PaddingType NNAPIConvert::getPaddingType(PaddingCode type)
{
  switch (type)
  {
    case ANEURALNETWORKS_PADDING_SAME:
      return PaddingType::SAME;
    case ANEURALNETWORKS_PADDING_VALID:
      return PaddingType::VALID;
    default:
      throw std::runtime_error("Unsupported type");
  }
}
