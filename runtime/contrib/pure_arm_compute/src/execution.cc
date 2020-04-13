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

#include "compilation.h"
#include "execution.h"
#include "profiling/profiling.h"
#include "profiling/profiler.h"
#include "event.h"

#include "internal/VectorSource.h"
#include "internal/MatrixSource.h"
#include "internal/Tensor3DSource.h"
#include "internal/FeatureSource.h"
#include "internal/TensorSource.h"

#include "internal/Sinks.h"
#include "internal/VectorSink.h"
#include "internal/MatrixSink.h"
#include "internal/Tensor3DSink.h"
#include "internal/FeatureSink.h"

#include "misc/feature/IndexIterator.h"

#include <arm_compute/runtime/CL/CLScheduler.h>

#include <cassert>

static void asVectorSource(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                           int32_t len, const void *buffer, size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->source<VectorSource<float>>(index, len, reinterpret_cast<const float *>(buffer),
                                             length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->source<VectorSource<int32_t>>(index, len,
                                               reinterpret_cast<const int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->source<VectorSource<uint32_t>>(index, len,
                                                reinterpret_cast<const uint32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->source<VectorSource<uint8_t>>(index, len,
                                               reinterpret_cast<const uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asMatrixSource(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                           const nnfw::misc::matrix::Shape &shape, const void *buffer,
                           size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->source<MatrixSource<float>>(index, shape, reinterpret_cast<const float *>(buffer),
                                             length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->source<MatrixSource<int32_t>>(index, shape,
                                               reinterpret_cast<const int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->source<MatrixSource<uint32_t>>(index, shape,
                                                reinterpret_cast<const uint32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->source<MatrixSource<uint8_t>>(index, shape,
                                               reinterpret_cast<const uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asTensor3DSource(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                             const nnfw::misc::tensor::Shape &shape, const void *buffer,
                             size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->source<Tensor3DSource<float>>(index, shape,
                                               reinterpret_cast<const float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->source<Tensor3DSource<int32_t>>(index, shape,
                                                 reinterpret_cast<const int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->source<Tensor3DSource<uint32_t>>(
          index, shape, reinterpret_cast<const uint32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->source<Tensor3DSource<uint8_t>>(index, shape,
                                                 reinterpret_cast<const uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asTensorSource(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                           const nnfw::misc::tensor::Shape &shape, const void *buffer,
                           size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->source<TensorSource<float>>(index, shape, reinterpret_cast<const float *>(buffer),
                                             length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->source<TensorSource<int32_t>>(index, shape,
                                               reinterpret_cast<const int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->source<TensorSource<uint32_t>>(index, shape,
                                                reinterpret_cast<const uint32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->source<TensorSource<uint8_t>>(index, shape,
                                               reinterpret_cast<const uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asFeatureSource(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                            const nnfw::misc::feature::Shape &shape, const void *buffer,
                            size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->source<FeatureSource<float>>(index, shape, reinterpret_cast<const float *>(buffer),
                                              length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->source<FeatureSource<int32_t>>(index, shape,
                                                reinterpret_cast<const int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->source<FeatureSource<uint32_t>>(
          index, shape, reinterpret_cast<const uint32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->source<FeatureSource<uint8_t>>(index, shape,
                                                reinterpret_cast<const uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asVectorSink(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                         int32_t len, void *buffer, size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->sink<VectorSink<float>>(index, len, reinterpret_cast<float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->sink<VectorSink<int32_t>>(index, len, reinterpret_cast<int32_t *>(buffer), length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->sink<VectorSink<uint32_t>>(index, len, reinterpret_cast<uint32_t *>(buffer),
                                            length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->sink<VectorSink<uint8_t>>(index, len, reinterpret_cast<uint8_t *>(buffer), length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asMatrixSink(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                         int32_t H, int32_t W, void *buffer, size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->sink<MatrixSink<float>>(index, H, W, reinterpret_cast<float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->sink<MatrixSink<int32_t>>(index, H, W, reinterpret_cast<int32_t *>(buffer),
                                           length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->sink<MatrixSink<uint32_t>>(index, H, W, reinterpret_cast<uint32_t *>(buffer),
                                            length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->sink<MatrixSink<uint8_t>>(index, H, W, reinterpret_cast<uint8_t *>(buffer),
                                           length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asFeatureSink(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                          const nnfw::misc::feature::Shape &shape, void *buffer, size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->sink<FeatureSink<float>>(index, shape, reinterpret_cast<float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->sink<FeatureSink<int32_t>>(index, shape, reinterpret_cast<int32_t *>(buffer),
                                            length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->sink<FeatureSink<uint32_t>>(index, shape, reinterpret_cast<uint32_t *>(buffer),
                                             length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->sink<FeatureSink<uint8_t>>(index, shape, reinterpret_cast<uint8_t *>(buffer),
                                            length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asTensor3DSink(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                           const nnfw::misc::tensor::Shape &shape, void *buffer, size_t length)
{
  assert(shape.rank() == 3);

  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->sink<Tensor3DSink<float>>(index, shape, reinterpret_cast<float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->sink<Tensor3DSink<int32_t>>(index, shape, reinterpret_cast<int32_t *>(buffer),
                                             length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->sink<Tensor3DSink<uint32_t>>(index, shape, reinterpret_cast<uint32_t *>(buffer),
                                              length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->sink<Tensor3DSink<uint8_t>>(index, shape, reinterpret_cast<uint8_t *>(buffer),
                                             length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

static void asTensorSink(ANeuralNetworksExecution *execution, int32_t type, int32_t index,
                         const nnfw::misc::tensor::Shape &shape, void *buffer, size_t length)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      execution->sink<TensorSink<float>>(index, shape, reinterpret_cast<float *>(buffer), length);
      break;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      execution->sink<TensorSink<int32_t>>(index, shape, reinterpret_cast<int32_t *>(buffer),
                                           length);
      break;
    case ANEURALNETWORKS_UINT32:
      execution->sink<TensorSink<uint32_t>>(index, shape, reinterpret_cast<uint32_t *>(buffer),
                                            length);
      break;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      execution->sink<TensorSink<uint8_t>>(index, shape, reinterpret_cast<uint8_t *>(buffer),
                                           length);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

//
// NNAPI Implementation
//
int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                    ANeuralNetworksExecution **execution)
{
  if ((compilation == nullptr) || (execution == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  std::shared_ptr<const ::internal::arm_compute::Plan> plan;
  compilation->publish(plan);
  ANeuralNetworksExecution *execution_ptr = new ANeuralNetworksExecution{plan};
  if (execution_ptr == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *execution = execution_ptr;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                      const ANeuralNetworksOperandType *type, const void *buffer,
                                      size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();

  // TODO Check type conflicts

  // NOTE The current implemenation assumes that every input is a feature map.
  // TODO Remove this assumption
  const auto operand_index = execution->plan().model().inputs.at(index);
  int32_t input_type = operands.at(operand_index).type();
  // NOTE TFLite passes type parameter unconditionally as nullptr.
  // Is it necessary to reget type value already set in model step?
  if (type != nullptr)
  {
    input_type = type->type;
  }

  auto shape = operands.at(operand_index).shape();
  auto rank = shape.rank();

  if (rank == 1)
  {
    const auto len = shape.dim(0);

    asVectorSource(execution, input_type, index, len, buffer, length);
  }
  else if (rank == 2)
  {
    const auto &operand_shape = shape.asMatrix();

    asMatrixSource(execution, input_type, index, operand_shape, buffer, length);
  }
  else if (rank == 3)
  {
    const auto &operand_shape = shape.asTensor();

    asTensor3DSource(execution, input_type, index, operand_shape, buffer, length);
  }
  else if (rank == 4)
  {
    const auto &operand_shape = shape.asFeature();

    asFeatureSource(execution, input_type, index, operand_shape, buffer, length);
  }
  else
  {
    // NOTE TensorSource is much slower than specialized Source(s)
    const auto &operand_shape = shape.asTensor();

    asTensorSource(execution, input_type, index, operand_shape, buffer, length);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

// squeeze(shape) eliminates all the dimensions whose dimensionality is 1
// For example, squeeze([3, 1, 3]) returns [3, 3]
static nnfw::misc::tensor::Shape squeeze(const nnfw::misc::tensor::Shape &shape)
{
  nnfw::misc::tensor::Shape res(0);

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    if (shape.dim(axis) != 1)
    {
      res.append(shape.dim(axis));
    }
  }

  return res;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                       const ANeuralNetworksOperandType *type, void *buffer,
                                       size_t length)
{
  // Don't check type
  // Comment about ANeuralNetworksOperandType in NeuralNetworks.h:
  //  If the input or output is optional and omitted then it need not have a fully specified tensor
  //  operand type
  if ((execution == nullptr) || ((buffer == nullptr) && (length != 0)))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();

  // TODO Check type conflicts

  const auto operand_index = execution->plan().model().outputs.at(index);
  int32_t output_type = operands.at(operand_index).type();
  const auto &output_shape = operands.at(operand_index).shape();

  if (output_shape.rank() == 1)
  {
    const auto len = output_shape.dim(0);

    asVectorSink(execution, output_type, index, len, buffer, length);
  }
  else if (output_shape.rank() == 2)
  {
    const auto H = output_shape.dim(0);
    const auto W = output_shape.dim(1);

    asMatrixSink(execution, output_type, index, H, W, buffer, length);
  }
  else if (output_shape.rank() == 3)
  {
    asTensor3DSink(execution, output_type, index, output_shape, buffer, length);
  }
  else if ((output_shape.rank() == 4))
  {
    const auto &operand_shape = operands.at(operand_index).shape().asFeature();

    asFeatureSink(execution, output_type, index, operand_shape, buffer, length);
  }
  else
  {
    // NOTE TensorSink is much slower than specialized Sink(s)
    const auto &shape = operands.at(operand_index).shape();
    asTensorSink(execution, output_type, index, shape, buffer, length);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                          ANeuralNetworksEvent **event)
{
  if ((execution == nullptr) || (event == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // TODO: Handle event
  ANeuralNetworksEvent *event_ptr = new ANeuralNetworksEvent{};
  if (event_ptr == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *event = event_ptr;

  return ANeuralNetworksExecution_compute(execution);
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution *execution)
{
  if (execution == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const bool sync = profiling::Context::get().sync();
  const auto &plan = execution->plan();
  const auto &model = plan.model();

  // Set input(s)
  for (uint32_t n = 0; n < model.inputs.size(); ++n)
  {
    auto setter = [&](::arm_compute::ITensor &tensor) { execution->source(n).push(tensor); };

    // Some operand may not be defined at plan. Because some operands
    // may be useless at ACL (ex. shape tensor for Reshape operator)
    // So added a sanity check.
    if (plan.operands().exist(model.inputs.at(n)))
    {
      plan.operands().at(model.inputs.at(n)).access(setter);
    }
  }

  const auto &operations = execution->plan().operations();

  for (uint32_t n = 0; n < operations.size(); ++n)
  {
    auto prof = profiling::Context::get().getProfiler();
    SCOPED_OPERATOR_PROFILE(prof, operations.at(n).op_idx());
    operations.at(n).run();

    if (sync)
    {
      arm_compute::CLScheduler::get().sync();
    }
  }

  // Get output(s)
  for (uint32_t n = 0; n < model.outputs.size(); ++n)
  {
    auto getter = [&](::arm_compute::ITensor &tensor) { execution->sink(n).pull(tensor); };

    plan.operands().at(model.outputs.at(n)).access(getter);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution *execution) {}

// TODO: implement this. added to fix link error on test build.
int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                const ANeuralNetworksOperandType *type,
                                                const ANeuralNetworksMemory *memory, size_t offset,
                                                size_t length)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  assert(false);
  return -1;
}

// TODO: implement this. added to fix link error on test build.
int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution *execution, int32_t index,
                                                 const ANeuralNetworksOperandType *type,
                                                 const ANeuralNetworksMemory *memory, size_t offset,
                                                 size_t length)
{
  if ((execution == nullptr) || (memory == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  assert(false);
  return -1;
}

int ANeuralNetworksExecution_getOutputOperandRank(ANeuralNetworksExecution *execution,
                                                  int32_t index, uint32_t *rank)
{
  if ((execution == nullptr) || (rank == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();
  const auto operand_index = execution->plan().model().outputs.at(index);
  const auto &output_shape = operands.at(operand_index).shape();

  *rank = output_shape.rank();

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_getOutputOperandDimensions(ANeuralNetworksExecution *execution,
                                                        int32_t index, uint32_t *dimensions)
{
  if ((execution == nullptr) || (dimensions == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  const auto &operands = execution->plan().model().operands();
  const auto operand_index = execution->plan().model().outputs.at(index);
  const auto &output_shape = operands.at(operand_index).shape();

  for (uint32_t axis = 0; axis < output_shape.rank(); ++axis)
  {
    dimensions[axis] = static_cast<uint32_t>(output_shape.dim(axis));
  }

  return ANEURALNETWORKS_NO_ERROR;
}
