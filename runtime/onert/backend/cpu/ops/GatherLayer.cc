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

#include "GatherLayer.h"

#include "OperationUtils.h"
#include "GGMLHelper.h"

#include <cker/operation/Gather.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void GatherLayer::configure(const IPortableTensor *input, const IPortableTensor *indices,
                            IPortableTensor *output, int32_t axis, ExternalContext *ctx)
{
  _input = input;
  _indices = indices;
  _axis = axis;
  _output = output;
  _ctx = ctx;

  if (_input->data_type() == OperandType::QUANT_GGML_Q4_0)
    ctx->initGgmlContext();
}

template <typename InputType> void GatherLayer::runByInputType()
{
  using OutputType = InputType;
  nnfw::cker::GatherParams op_params;
  op_params.axis = _axis;

  switch (_indices->data_type())
  {
    case OperandType::INT32:
    {
      using IndicesType = int32_t;

      nnfw::cker::Gather<InputType, IndicesType>(
        op_params, getShape(_input), getBuffer<InputType>(_input), getShape(_indices),
        getBuffer<IndicesType>(_indices), getShape(_output), getBuffer<OutputType>(_output));
      break;
    }
    case OperandType::INT64:
    {
      using IndicesType = int64_t;

      nnfw::cker::Gather<InputType, IndicesType>(
        op_params, getShape(_input), getBuffer<InputType>(_input), getShape(_indices),
        getBuffer<IndicesType>(_indices), getShape(_output), getBuffer<OutputType>(_output));
      break;
    }
    default:
      throw std::runtime_error("Gather: unsupported indices data type");
  }
}

void GatherLayer::runByGGMLQuantInputType()
{
  // Supporting condition
  // Input: rank 2
  // Indice: rank < 4 or rank 4 with dim(0) = 1, INT32
  // Axis: 0
  if (getShape(_input).DimensionsCount() != 2)
    throw std::runtime_error("Gather: block quantized input tensor must be rank 2");

  if (getShape(_indices).DimensionsCount() >= 4 &&
      (getShape(_indices).DimensionsCount() != 4 || getShape(_indices).Dims(0) != 1))
    throw std::runtime_error("Gather: invalid indices tensor shape");

  if (_indices->data_type() != ir::DataType::INT32)
    throw std::runtime_error("Gather: indices tensor must be int32 type");

  if (_axis != 0)
    throw std::runtime_error("Gather: axis must be 0");

  // convert tensor
  auto input = getGGMLTensor(_input);
  auto indices = getGGMLTensor(_indices);
  auto output = getGGMLTensor(_output);
  {
    output.op = GGML_OP_GET_ROWS;
    output.src[0] = &input;
    output.src[1] = &indices;
  }
  auto *nodes = &output;

  // create graph
  struct ggml_cgraph graph;
  {
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = 1;
    graph.nodes = &nodes;
  }

  // get cplan
  auto cplan = ggml_graph_plan(&graph, _ctx->maxNumThreads());
  std::vector<uint8_t> buf(cplan.work_size);
  cplan.work_data = buf.data();

  // compute
  ggml_graph_compute(&graph, &cplan);
}

void GatherLayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      runByInputType<float>();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      runByInputType<uint8_t>();
      break;
    case OperandType::INT32:
      runByInputType<int32_t>();
      break;
    case OperandType::QUANT_GGML_Q4_0: {
      runByGGMLQuantInputType();
      auto newShape = _output->getShape();
      newShape.extendRank(4);
      _output->setShape(newShape);
      break;
    }
    default:
      throw std::runtime_error("Gather: unsupported input data type");
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
