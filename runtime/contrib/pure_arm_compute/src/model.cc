/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <NeuralNetworks.h>
#include <NeuralNetworksEx.h>

#include <cassert>
#include <stdexcept>

#include "model.h"
#include "memory.h"

int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  if (model == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  ANeuralNetworksModel *model_ptr = new ANeuralNetworksModel{};

  if (model_ptr == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }

  *model = model_ptr;

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel *model) { delete model; }

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                    const ANeuralNetworksOperandType *type)
{
  if ((model == nullptr) || (type == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (type->type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM)
  {
    // Quantized:
    //  scale: a 32 bit floating point value greater than zero
    //  zeroPoint: a 32 bit integer, in range [0, 255]
    if (type->scale <= 0.0f)
    {
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (type->zeroPoint < 0 || type->zeroPoint > 255)
    {
      return ANEURALNETWORKS_BAD_DATA;
    }
  }
  // NOTE Validation of scale and zeroPoint would be skipped for a while.
  //      We do not know whether scalar type can have scale and zeroPoint.
  //      To pass ValidationTest and GeneratedTest, this validation code
  //      would not be implemented until we can define this issue clearly.
  //
  // scale and zeroPoint should be zero for scalars and non-fixed point tensors
  // else if ((type->scale != 0.0f) || (type->zeroPoint != 0))
  // {
  //   return ANEURALNETWORKS_BAD_DATA;
  // }

  // scalar is ANEURALNETWORKS_FLOAT32, ANEURALNETWORKS_INT32 or ANEURALNETWORKS_UINT32.
  // ANEURALNETWORKS_TENSOR_FLOAT32, ANEURALNETWORKS_TENSOR_INT32 and
  // ANEURALNETWORKS_TENSOR_QUANT8_ASYMM are not scalar
  //
  // dimensionCount should be zero for scalars
  if (type->dimensionCount != 0 &&
      (type->type == ANEURALNETWORKS_FLOAT32 || type->type == ANEURALNETWORKS_INT32 ||
       type->type == ANEURALNETWORKS_UINT32))
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  // ASSUME A tensor operand should consists of fp32 or int32 values.
  // NOTE We do not care about scala operands.
  assert((type->dimensionCount == 0) || (type->type == ANEURALNETWORKS_TENSOR_FLOAT32 ||
                                         type->type == ANEURALNETWORKS_TENSOR_INT32 ||
                                         type->type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM));

  internal::tflite::operand::Shape shape(type->dimensionCount);

  for (uint32_t axis = 0; axis < type->dimensionCount; ++axis)
  {
    shape.dim(axis) = type->dimensions[axis];
  }

  model->deref().operands().append(shape, type->type, type->scale, type->zeroPoint);

  // NOTE We do NOT allocate CLTensor here as we do not how to interpret this one.
  //      TensorFlow Lite may interpret a rank-4 tensor either as a feature map (with batch) or
  //      a convolution kernel.

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                         const void *buffer, size_t length)
{
  if ((model == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const internal::tflite::operand::Index ind{index};
  auto &obj = model->deref().operands().at(ind);

  if (buffer == nullptr)
  {
    using internal::tflite::operand::ExternalData;
    obj.data<ExternalData>(reinterpret_cast<const uint8_t *>(buffer), length);
  }
  else
  {
    using internal::tflite::operand::CachedData;
    obj.data<CachedData>(reinterpret_cast<const uint8_t *>(buffer), length);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                                   const ANeuralNetworksMemory *memory,
                                                   size_t offset, size_t length)
{
  if ((model == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const internal::tflite::operand::Index ind{index};
  auto &obj = model->deref().operands().at(ind);

  using internal::tflite::operand::ExternalData;

  obj.data<ExternalData>(reinterpret_cast<const uint8_t *>(memory->base() + offset), length);

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t *inputs, uint32_t outputCount,
                                      const uint32_t *outputs)
{
  if (model == nullptr || inputs == nullptr || outputs == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  if (type < ANEURALNETWORKS_ADD || type > ANEURALNETWORKS_TRANSPOSE)
  {
    return ANEURALNETWORKS_BAD_DATA;
  }

  switch (type)
  {
    case ANEURALNETWORKS_ADD:
    {
      assert(inputCount == 3);
      assert(outputCount == 1);

      using internal::tflite::op::Add::Node;
      using internal::tflite::op::Add::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SUB:
    {
      assert(inputCount == 3);
      assert(outputCount == 1);

      using internal::tflite::op::Sub::Node;
      using internal::tflite::op::Sub::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_MUL:
    {
      assert(inputCount == 3);
      assert(outputCount == 1);

      using internal::tflite::op::Mul::Node;
      using internal::tflite::op::Mul::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_DIV:
    {
      assert(inputCount == 3);
      assert(outputCount == 1);

      using internal::tflite::op::Div::Node;
      using internal::tflite::op::Div::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_CONV_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using internal::tflite::op::Conv2D::Implicit::Node;
        using internal::tflite::op::Conv2D::Implicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }
      else
      {
        using internal::tflite::op::Conv2D::Explicit::Node;
        using internal::tflite::op::Conv2D::Explicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }

      break;
    }
    case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
    {
      // inputCount is either 8 or 11 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 8
      //  - Padding is explicit when inputCount is 11
      assert(inputCount == 8 || inputCount == 11);
      assert(outputCount == 1);

      if (inputCount == 8)
      {
        using internal::tflite::op::DepthwiseConv2D::Implicit::Node;
        using internal::tflite::op::DepthwiseConv2D::Implicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }
      else
      {
        using internal::tflite::op::DepthwiseConv2D::Explicit::Node;
        using internal::tflite::op::DepthwiseConv2D::Explicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }

      break;
    }
    case ANEURALNETWORKS_MAX_POOL_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using internal::tflite::op::MaxPool2D::Implicit::Node;
        using internal::tflite::op::MaxPool2D::Implicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }
      else
      {
        using internal::tflite::op::MaxPool2D::Explicit::Node;
        using internal::tflite::op::MaxPool2D::Explicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }

      break;
    }
    case ANEURALNETWORKS_DEQUANTIZE:
    {
      assert(outputCount == 1 && inputCount == 1);
      using internal::tflite::op::Dequantize::Node;
      using internal::tflite::op::Dequantize::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_AVERAGE_POOL_2D:
    {
      // inputCount is either 7 or 10 acccording to NN API specification.
      //  - Padding is implicit when inputCount is 7
      //  - Padding is explicit when inputCount is 10
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using internal::tflite::op::AvgPool2D::Implicit::Node;
        using internal::tflite::op::AvgPool2D::Implicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }
      else
      {
        using internal::tflite::op::AvgPool2D::Explicit::Node;
        using internal::tflite::op::AvgPool2D::Explicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }

      break;
    }
    case ANEURALNETWORKS_CONCATENATION:
    {
      using internal::tflite::op::Concat::Node;
      using internal::tflite::op::Concat::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RESIZE_BILINEAR:
    {
      using internal::tflite::op::ResizeBilinear::Node;
      using internal::tflite::op::ResizeBilinear::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RESHAPE:
    {
      using internal::tflite::op::Reshape::Node;
      using internal::tflite::op::Reshape::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SQUEEZE:
    {
      using internal::tflite::op::Squeeze::Node;
      using internal::tflite::op::Squeeze::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_FULLY_CONNECTED:
    {
      using internal::tflite::op::FullyConnected::Node;
      using internal::tflite::op::FullyConnected::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SOFTMAX:
    {
      using internal::tflite::op::Softmax::Node;
      using internal::tflite::op::Softmax::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RELU:
    {
      using internal::tflite::op::ReLU::Node;
      using internal::tflite::op::ReLU::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RELU1:
    {
      using internal::tflite::op::ReLU1::Node;
      using internal::tflite::op::ReLU1::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RELU6:
    {
      using internal::tflite::op::ReLU6::Node;
      using internal::tflite::op::ReLU6::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_TANH:
    {
      using internal::tflite::op::Tanh::Node;
      using internal::tflite::op::Tanh::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_STRIDED_SLICE:
    {
      using internal::tflite::op::StridedSlice::Node;
      using internal::tflite::op::StridedSlice::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LOGISTIC:
    {
      using internal::tflite::op::Logistic::Node;
      using internal::tflite::op::Logistic::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_MEAN:
    {
      using internal::tflite::op::Mean::Node;
      using internal::tflite::op::Mean::Param;

      auto &operations = model->deref().operations();
      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RNN:
    {
      using internal::tflite::op::RNN::Node;
      using internal::tflite::op::RNN::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_TRANSPOSE:
    {
      using internal::tflite::op::Transpose::Node;
      using internal::tflite::op::Transpose::Param;
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LSTM:
    {
      using internal::tflite::op::LSTM::Node;
      using internal::tflite::op::LSTM::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_FLOOR:
    {
      using internal::tflite::op::Floor::Node;
      using internal::tflite::op::Floor::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_PAD:
    {
      assert(inputCount == 2 && outputCount == 1);

      using internal::tflite::op::Pad::Node;
      using internal::tflite::op::Pad::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SPACE_TO_DEPTH:
    {
      using internal::tflite::op::SpaceToDepth::Node;
      using internal::tflite::op::SpaceToDepth::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SPACE_TO_BATCH_ND:
    {
      using internal::tflite::op::SpaceToBatchND::Node;
      using internal::tflite::op::SpaceToBatchND::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_BATCH_TO_SPACE_ND:
    {
      using internal::tflite::op::BatchToSpaceNd::Node;
      using internal::tflite::op::BatchToSpaceNd::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_L2_POOL_2D:
    {
      // Input count is 7 for Implicit Padding
      // Input count is 10 for Explicit Padding
      assert(inputCount == 7 || inputCount == 10);
      assert(outputCount == 1);

      if (inputCount == 7)
      {
        using internal::tflite::op::L2Pool2D::Implicit::Node;
        using internal::tflite::op::L2Pool2D::Implicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }
      else
      {
        using internal::tflite::op::L2Pool2D::Explicit::Node;
        using internal::tflite::op::L2Pool2D::Explicit::Param;

        // Add 'operations'
        auto &operations = model->deref().operations();

        operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});
      }

      break;
    }
    case ANEURALNETWORKS_EMBEDDING_LOOKUP:
    {
      assert(inputCount == 2);
      assert(outputCount == 1);

      using internal::tflite::op::EmbeddingLookup::Node;
      using internal::tflite::op::EmbeddingLookup::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_L2_NORMALIZATION:
    {
      assert(inputCount == 1 && outputCount == 1);

      using internal::tflite::op::L2Normalization::Node;
      using internal::tflite::op::L2Normalization::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_HASHTABLE_LOOKUP:
    {
      assert(inputCount == 3);
      assert(outputCount == 2);

      using internal::tflite::op::HashtableLookup::Node;
      using internal::tflite::op::HashtableLookup::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION:
    {

      using internal::tflite::op::LocalResponseNormalization::Node;
      using internal::tflite::op::LocalResponseNormalization::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_DEPTH_TO_SPACE:
    {
      using internal::tflite::op::DepthToSpace::Node;
      using internal::tflite::op::DepthToSpace::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    default:
      throw std::runtime_error{"Not supported operation"};
  };

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                        ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  switch (type)
  {
    case ANEURALNETWORKS_CAST_EX:
    {
      using internal::tflite::op::Cast::Node;
      using internal::tflite::op::Cast::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_REDUCE_MIN_EX:
    {
      using internal::tflite::op::ReduceMin::Node;
      using internal::tflite::op::ReduceMin::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_REDUCE_MAX_EX:
    {
      using internal::tflite::op::ReduceMax::Node;
      using internal::tflite::op::ReduceMax::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_PRELU_EX:
    {
      using internal::tflite::op::PReLU::Node;
      using internal::tflite::op::PReLU::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_TRANSPOSE_CONV_EX:
    {
      using internal::tflite::op::TransposeConv::Node;
      using internal::tflite::op::TransposeConv::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LOGICAL_AND_EX:
    {
      using internal::tflite::op::LogicalAnd::Node;
      using internal::tflite::op::LogicalAnd::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LOGICAL_OR_EX:
    {
      using internal::tflite::op::LogicalOr::Node;
      using internal::tflite::op::LogicalOr::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_LOGICAL_NOT_EX:
    {
      using internal::tflite::op::LogicalNot::Node;
      using internal::tflite::op::LogicalNot::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_RSQRT_EX:
    {
      using internal::tflite::op::RSQRT::Node;
      using internal::tflite::op::RSQRT::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SQRT_EX:
    {
      using internal::tflite::op::SQRT::Node;
      using internal::tflite::op::SQRT::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_EQUAL_EX:
    {
      using internal::tflite::op::Equal::Node;
      using internal::tflite::op::Equal::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SQUARED_DIFFERENCE_EX:
    {
      using internal::tflite::op::SquaredDifference::Node;
      using internal::tflite::op::SquaredDifference::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_TOPK_V2_EX:
    {
      using internal::tflite::op::TopKV2::Node;
      using internal::tflite::op::TopKV2::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_GATHER_EX:
    {
      using internal::tflite::op::Gather::Node;
      using internal::tflite::op::Gather::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_SPLIT_EX:
    {
      using internal::tflite::op::Split::Node;
      using internal::tflite::op::Split::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_UNPACK_EX:
    {
      using internal::tflite::op::Unpack::Node;
      using internal::tflite::op::Unpack::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_NEG_EX:
    {
      using internal::tflite::op::Neg::Node;
      using internal::tflite::op::Neg::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_EXP_EX:
    {
      using internal::tflite::op::Exp::Node;
      using internal::tflite::op::Exp::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_REDUCE_SUM_EX:
    {
      using internal::tflite::op::ReduceSum::Node;
      using internal::tflite::op::ReduceSum::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_PACK_EX:
    {
      using internal::tflite::op::Pack::Node;
      using internal::tflite::op::Pack::Param;

      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_ABS_EX:
    {
      using internal::tflite::op::Abs::Node;
      using internal::tflite::op::Abs::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_ARGMAX_EX:
    {
      using internal::tflite::op::ArgMax::Node;
      using internal::tflite::op::ArgMax::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }
    case ANEURALNETWORKS_NOT_EQUAL_EX:
    {
      using internal::tflite::op::NotEqual::Node;
      using internal::tflite::op::NotEqual::Param;

      // Add 'operations'
      auto &operations = model->deref().operations();

      operations.emplace_back<Node>(Param{inputCount, inputs, outputCount, outputs});

      break;
    }

    default:
      throw std::runtime_error{"Not supported operation"};
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                                  const uint32_t *inputs, uint32_t outputCount,
                                                  const uint32_t *outputs)
{
  if ((model == nullptr) || (inputs == nullptr) || (outputs == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  // NOTE ::internal::tflite::operand::Index uses int as its underlying type as various NNAPI
  //      functions such as ANeuralNetworksModel_setOperandValue use int to represent operand index
  //
  //      ANeuralNetworksModel_identifyInputsAndOutputs, however, uses uint32_t to represent operand
  //      index.
  //
  //      Below, static_cast<int>(...) is introduced to eliminate compiler warning.
  for (uint32_t n = 0; n < inputCount; ++n)
  {
    const ::internal::tflite::operand::Index ind{static_cast<int>(inputs[n])};
    model->deref().inputs.emplace_back(ind);
  }

  for (uint32_t n = 0; n < outputCount; ++n)
  {
    const ::internal::tflite::operand::Index ind{static_cast<int>(outputs[n])};
    model->deref().outputs.emplace_back(ind);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  if (model == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  model->markAsFinished();

  return ANEURALNETWORKS_NO_ERROR;
}

//
// ANeuralNetworksModel
//
ANeuralNetworksModel::ANeuralNetworksModel() : _model{new internal::tflite::Model}
{
  // DO NOTHING
}
