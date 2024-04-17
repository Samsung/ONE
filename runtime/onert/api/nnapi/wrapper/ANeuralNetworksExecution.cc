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

#include "ANeuralNetworksExecution.h"
#include "NNAPIConvert.h"
#include "util/logging.h"

const onert::ir::OperandIndex ANeuralNetworksExecution::getInputOperandIndex(int32_t index) noexcept
{
  if (index < 0)
  {
    // Negative index: return invalid index
    return onert::ir::OperandIndex{};
  }

  uint32_t cast_index = static_cast<uint32_t>(index);
  if (cast_index >= _execution->primary_subgraph().getInputs().size())
  {
    // Return invalid index
    return onert::ir::OperandIndex{};
  }

  onert::ir::IOIndex input_index{cast_index};
  const auto operand_index = _execution->primary_subgraph().getInputs().at(input_index);
  return operand_index;
}

const onert::ir::OperandIndex
ANeuralNetworksExecution::getOutputOperandIndex(int32_t index) noexcept
{
  if (index < 0)
  {
    // Negative index: return invalid index
    return onert::ir::OperandIndex{};
  }

  uint32_t cast_index = static_cast<uint32_t>(index);
  if (cast_index >= _execution->primary_subgraph().getOutputs().size())
  {
    // Return invalid index
    return onert::ir::OperandIndex{};
  }

  onert::ir::IOIndex output_index{cast_index};
  const auto operand_index = _execution->primary_subgraph().getOutputs().at(output_index);
  return operand_index;
}

bool ANeuralNetworksExecution::compareDataType(const ANeuralNetworksOperandType *type,
                                               const onert::ir::OperandIndex index) noexcept
{
  try
  {
    const auto &operand_type = _execution->primary_subgraph().operands().at(index).typeInfo();
    const auto typeInfo = NNAPIConvert::getTypeInfo(type);

    if (operand_type != typeInfo)
    {
      // Data type mismatch
      return false;
    }
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::compareShape(const ANeuralNetworksOperandType *type,
                                            const onert::ir::OperandIndex index) noexcept
{
  // Passed shape should be specified
  if (hasUnspecifiedDims(index))
  {
    return false;
  }

  const auto &operand_shape = _execution->primary_subgraph().operands().at(index).shape();
  const auto &shape_from_type = NNAPIConvert::getShape(type);

  return operand_shape == shape_from_type;
}

bool ANeuralNetworksExecution::IsOptionalInput(const onert::ir::OperandIndex index) noexcept
{
  const auto &operand_shape = _execution->primary_subgraph().operands().at(index).shape();
  for (int32_t i = 0; i < operand_shape.rank(); ++i)
  {
    if (operand_shape.dim(i) != 0)
      return false;
  }
  return true;
}

bool ANeuralNetworksExecution::hasUnspecifiedDims(const onert::ir::OperandIndex index) noexcept
{
  const auto &operand_shape = _execution->primary_subgraph().operands().at(index).shape();

  return operand_shape.hasUnspecifiedDims();
}

size_t ANeuralNetworksExecution::getOperandSize(const onert::ir::OperandIndex index) noexcept
{
  try
  {
    return _execution->primary_subgraph().operands().at(index).operandSize();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return 0;
  }
}

bool ANeuralNetworksExecution::setInput(uint32_t index, const ANeuralNetworksOperandType *type,
                                        const void *buffer, size_t length) noexcept
{
  try
  {
    onert::ir::IOIndex input_index{index};
    const auto shape =
      (type != nullptr) ? NNAPIConvert::getShape(type) : _execution->getInputShape(input_index);

    _execution->setInput(input_index, shape, buffer, length);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::setOptionalInput(uint32_t index,
                                                const ANeuralNetworksOperandType *type,
                                                const void *buffer, size_t length) noexcept
{
  assert(type == nullptr);
  assert(buffer == nullptr);
  assert(length == 0);
  try
  {
    onert::ir::IOIndex input_index{index};
    const auto shape =
      (type != nullptr) ? NNAPIConvert::getShape(type) : _execution->getInputShape(input_index);

    // ANeuralNetworksExecution::setInput() uses only shape information
    ANeuralNetworksOperandType optional_input_type;
    optional_input_type.dimensionCount = shape.rank();
    std::vector<uint32_t> dims(optional_input_type.dimensionCount);
    for (uint32_t i = 0; i < optional_input_type.dimensionCount; ++i)
    {
      dims.at(i) = shape.dim(i);
    }
    optional_input_type.dimensions = dims.data();

    return setInput(index, &optional_input_type, buffer, length);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::setOutput(uint32_t index, const ANeuralNetworksOperandType *type,
                                         void *buffer, size_t length) noexcept
{
  try
  {
    onert::ir::IOIndex output_index{index};
    const auto shape =
      (type != nullptr) ? NNAPIConvert::getShape(type) : _execution->getOutputShape(output_index);

    _execution->setOutput(output_index, shape, buffer, length);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::startExecute(void) noexcept
{
  try
  {
    _execution->startExecute();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::execute(void) noexcept
{
  try
  {
    _execution->execute();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

const std::shared_ptr<onert::exec::Execution> ANeuralNetworksExecution::instance(void) noexcept
{
  return _execution;
}

bool ANeuralNetworksExecution::getOutputOperandRank(uint32_t index, uint32_t *rank) noexcept
{
  try
  {
    onert::ir::IOIndex output_index{index};

    // Check execution is finished
    if (!_execution->isFinished())
    {
      return false;
    }

    const auto shape = _execution->getOutputShape(output_index);
    if (shape.hasUnspecifiedDims())
    {
      throw std::runtime_error{"Internal error: Output tensor has unspecified dims"};
    }

    *rank = shape.rank();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksExecution::getOutputOperandDimensions(uint32_t index, uint32_t *dimensions)
{
  try
  {
    onert::ir::IOIndex output_index{index};

    // Check execution is finished
    if (!_execution->isFinished())
    {
      return false;
    }

    const auto shape = _execution->getOutputShape(output_index);
    if (shape.hasUnspecifiedDims())
    {
      throw std::runtime_error{"Internal error: Output tensor has unspecified dims"};
    }

    for (int i = 0; i < shape.rank(); i++)
    {
      auto dim = shape.dim(i);

      if (dim <= 0)
      {
        throw std::runtime_error{"Invalid dimension value"};
      }

      dimensions[i] = static_cast<uint32_t>(dim);
    }
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}
