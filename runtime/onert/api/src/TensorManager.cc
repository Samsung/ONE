/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TensorManager.h"
#include "Helper.h"

namespace onert
{
namespace api
{

namespace
{

onert::ir::Layout convert_layout(NNFW_LAYOUT layout)
{
  if (layout == NNFW_LAYOUT_CHANNELS_LAST)
  {
    return onert::ir::Layout::NHWC;
  }
  else if (layout == NNFW_LAYOUT_CHANNELS_FIRST)
  {
    return onert::ir::Layout::NCHW;
  }
  return onert::ir::Layout::UNKNOWN;
}

static NNFW_TYPE datatype_to_nnfw_dtype(onert::ir::DataType dt)
{
  using onert::ir::DataType;
  switch (dt)
  {
    case DataType::FLOAT32:
      return NNFW_TYPE_TENSOR_FLOAT32;
    case DataType::INT32:
      return NNFW_TYPE_TENSOR_INT32;
    case DataType::QUANT_UINT8_ASYMM:
      return NNFW_TYPE_TENSOR_QUANT8_ASYMM;
    case DataType::BOOL8:
      return NNFW_TYPE_TENSOR_BOOL;
    case DataType::UINT8:
      return NNFW_TYPE_TENSOR_UINT8;
    case DataType::INT64:
      return NNFW_TYPE_TENSOR_INT64;
    case DataType::QUANT_INT8_ASYMM:
      return NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED;
    case DataType::UINT32:
    case DataType::QUANT_INT8_SYMM:
    default:
      throw std::runtime_error("Error: Model has type that runtime API does not support.");
  }
}

NNFW_STATUS getTensorIndexImpl(const onert::ir::Graph &graph, const char *tensorname,
                               uint32_t *index, bool is_input)
{
  if (!tensorname || !index)
    return NNFW_STATUS_UNEXPECTED_NULL;

  if (!null_terminating(tensorname, MAX_TENSOR_NAME_LENGTH))
  {
    std::cerr << "nnpackage path is too long" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  auto ind_found = is_input ? graph.getInputIndex(tensorname) : graph.getOutputIndex(tensorname);

  if (ind_found.undefined())
  {
    // Not found
    return NNFW_STATUS_ERROR;
  }
  else
  {
    *index = ind_found.value();
    return NNFW_STATUS_NO_ERROR;
  }
}

} // namespace

TensorManager::TensorManager(nnfw_session *session) : _session{session}
{
  // DO NOTHING
}

NNFW_STATUS TensorManager::setInput(uint32_t index, NNFW_TYPE /* type */, const void *buffer,
                                    size_t length)
{
  if (!_session->isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_input : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!buffer && length != 0)
  {
    std::cerr
      << "Error during nnfw_session::set_input : given buffer is NULL but the length is not 0"
      << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _session->_execution->setInput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::setOutput(uint32_t index, NNFW_TYPE /*type*/, void *buffer,
                                     size_t length)
{
  if (!_session->isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_output : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!buffer && length != 0)
  {
    std::cerr
      << "Error during nnfw_session::set_output : given buffer is NULL but the length is not 0"
      << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _session->_execution->setOutput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::inputSize(uint32_t *number)
{
  if (_session->isStateInitialized()) // Model is not loaded
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_size, number is null pointer." << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    *number = _session->primary_subgraph()->getInputs().size();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::input_size : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::outputSize(uint32_t *number)
{
  if (_session->isStateInitialized()) // Model is not loaded
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::output_size, number is null pointer." << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    *number = _session->primary_subgraph()->getOutputs().size();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_size" << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::setInputLayout(uint32_t index, NNFW_LAYOUT layout)
{
  try
  {
    if (layout != NNFW_LAYOUT_NONE && layout != NNFW_LAYOUT_CHANNELS_FIRST &&
        layout != NNFW_LAYOUT_CHANNELS_LAST)
    {
      std::cerr << "Error during nnfw_session::set_input_layout, not supported layout" << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _session->_execution->setInputLayout(onert::ir::IOIndex(index), convert_layout(layout));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::setOutputLayout(uint32_t index, NNFW_LAYOUT layout)
{
  try
  {
    if (layout != NNFW_LAYOUT_NONE && layout != NNFW_LAYOUT_CHANNELS_FIRST &&
        layout != NNFW_LAYOUT_CHANNELS_LAST)
    {
      std::cerr << "Error during nnfw_session::set_output_layout, not supported layout"
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _session->_execution->setOutputLayout(onert::ir::IOIndex(index), convert_layout(layout));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::applyTensorinfo(uint32_t index, nnfw_tensorinfo ti)
{
  // sanity check
  {
    if (_session->isStateInitialized())
    {
      std::cerr << "Error during set_input_tensorinfo : should be run after load_model"
                << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    if (ti.rank <= 0 || ti.rank > NNFW_MAX_RANK)
    {
      std::cerr << "unsupported rank: " << ti.rank << std::endl;
      return NNFW_STATUS_ERROR;
    }

    for (int32_t i = 0; i < ti.rank; ++i)
    {
      if (ti.dims[i] <= 0)
      {
        std::cerr << "dim must be positive integer but was " << ti.dims[i] << std::endl;
        return NNFW_STATUS_ERROR;
      }
    }
  }

  onert::ir::Shape new_shape(ti.rank);
  for (int32_t i = 0; i < ti.rank; i++)
    new_shape.dim(i) = ti.dims[i];

  if (!_session->isStatePreparedOrFinishedRun())
  {
    // In this case, if we apply input shape in primary_subgraph, it will propagate after
    // compilation and excution
    auto primary_subgraph = _session->_subgraphs->primary();
    auto ind = primary_subgraph->getInputs().at(index);
    auto &input = primary_subgraph->operands().at(ind);

    // overwrite input shape with the shape from ti
    input.info().shape(new_shape);
  }
  else // when called after nnfw_session::prepare()
  {
    _session->_execution->changeInputShape(onert::ir::IOIndex(index), new_shape);
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::setInputTensorinfo(uint32_t index, const nnfw_tensorinfo *ti)
{
  nnfw_tensorinfo ti_copy = *ti;
  return applyTensorinfo(index, ti_copy);
}

NNFW_STATUS TensorManager::inputTensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (_session->isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (ti == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, tensorinfo is null pointer."
                << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    if (index >= _session->primary_subgraph()->getInputs().size())
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = _session->primary_subgraph()->getInputs().at(index);
    auto shape = _session->primary_subgraph()->operands().at(opidx).shape();
    if (_session->isStatePreparedOrFinishedRun())
      shape = _session->_execution->getInputShape(onert::ir::IOIndex{index});
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype =
      datatype_to_nnfw_dtype(_session->primary_subgraph()->operands().at(opidx).typeInfo().type());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::input_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::outputTensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (_session->isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (ti == nullptr)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo, tensorinfo is null pointer."
              << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (index >= _session->primary_subgraph()->getOutputs().size())
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo, index is out of range."
              << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    auto opidx = _session->primary_subgraph()->getOutputs().at(index);
    auto shape = _session->primary_subgraph()->operands().at(opidx).shape();
    // If it is called after `nnfw_run` then get the shape from Execution, not from the graph
    if (_session->isStateFinishedRun())
      shape = _session->_execution->getOutputShape(onert::ir::IOIndex{index});
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype =
      datatype_to_nnfw_dtype(_session->primary_subgraph()->operands().at(opidx).typeInfo().type());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS TensorManager::inputTensorindex(const char *tensorname, uint32_t *index)
{
  return getTensorIndexImpl(*_session->primary_subgraph(), tensorname, index, true);
}

NNFW_STATUS TensorManager::outputTensorindex(const char *tensorname, uint32_t *index)
{
  return getTensorIndexImpl(*_session->primary_subgraph(), tensorname, index, false);
}

} // namespace api
} // namespace onert
