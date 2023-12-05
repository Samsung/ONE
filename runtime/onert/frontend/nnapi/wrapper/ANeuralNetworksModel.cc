/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ANeuralNetworksModel.h"
#include "OperationFactory.h"
#include "NNAPIConvert.h"

#include "ir/Operations.Include.h"
#include "util/logging.h"

#include <memory>

//
// ANeuralNetworksModel
//
ANeuralNetworksModel::ANeuralNetworksModel() noexcept
  : _finished_building{false}, _optional_operands{}, _operand_usages{}, _allowFloat32toFloat16{
                                                                          false}
{
  _graph = std::make_shared<onert::ir::Graph>();
}

bool ANeuralNetworksModel::addOperand(const ANeuralNetworksOperandType *type) noexcept
{
  try
  {
    const auto shape = NNAPIConvert::getShape(type);
    const auto typeInfo = NNAPIConvert::getTypeInfo(type);
    _graph->addOperand(shape, typeInfo);
    _operand_usages.emplace_back(OperandUsage::NOT_DEFINED);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksModel::setOperandValue(uint32_t index, const void *buffer, size_t length,
                                           bool optional, bool copy) noexcept
{
  const onert::ir::OperandIndex ind{index};

  try
  {
    _operand_usages[index] = OperandUsage::CONSTANT;

    // Remain operands.at(ind).data()->base() as nullptr for optional operand
    // This will be filled when model finished
    if (optional)
    {
      setOptionalOperand(ind);
    }

    using onert::ir::CachedData;
    using onert::ir::ExternalData;
    if (copy)
    {
      _graph->operands().at(ind).data(
        std::make_unique<CachedData>(reinterpret_cast<const uint8_t *>(buffer), length));
    }
    else
    {
      _graph->operands().at(ind).data(
        std::make_unique<ExternalData>(reinterpret_cast<const uint8_t *>(buffer), length));
    }
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksModel::addOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs) noexcept
{
  try
  {
    for (uint32_t i = 0; i < outputCount; i++)
    {
      _operand_usages[outputs[i]] = OperandUsage::OPERATION_OUTPUT;
    }

    auto &factory = OperationFactory::get();
    OperationFactory::Param param{inputCount, inputs, outputCount, outputs};

    auto node = factory.create(type, param, _graph->operands());
    _graph->addOperation(std::unique_ptr<onert::ir::Operation>{node});

    // TODO Move these codes to delegate.cpp
    if (type == ANEURALNETWORKS_FULLY_CONNECTED)
    {
      const auto &input_operand =
        _graph->operands().at(node->getInputs().at(onert::ir::operation::FullyConnected::INPUT));
      auto &weights_operand =
        _graph->operands().at(node->getInputs().at(onert::ir::operation::FullyConnected::WEIGHT));
      if (input_operand.typeInfo().type() == onert::ir::DataType::FLOAT32 &&
          weights_operand.typeInfo().type() == onert::ir::DataType::QUANT_UINT8_ASYMM)
      {
        weights_operand.type(onert::ir::DataType::QUANT_INT8_SYMM);
      }
    }

    // TODO Update operand layout
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

bool ANeuralNetworksModel::addOperationEx(ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                          const uint32_t *inputs, uint32_t outputCount,
                                          const uint32_t *outputs) noexcept
{
  try
  {
    for (uint32_t i = 0; i < outputCount; i++)
    {
      _operand_usages[outputs[i]] = OperandUsage::OPERATION_OUTPUT;
    }

    auto &factory = OperationFactory::get();
    OperationFactory::Param param{inputCount, inputs, outputCount, outputs};

    auto node = factory.create(type, param, _graph->operands());
    _graph->addOperation(std::unique_ptr<onert::ir::Operation>{node});
  }
  catch (const std::exception &e)
  {
    return false;
  }
  return true;
}

bool ANeuralNetworksModel::addModelInput(uint32_t index) noexcept
{
  try
  {
    _operand_usages[index] = OperandUsage::MODEL_INPUT;

    const onert::ir::OperandIndex ind{index};
    _graph->addInput(ind);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}
bool ANeuralNetworksModel::addModelOutput(uint32_t index) noexcept
{
  try
  {
    const onert::ir::OperandIndex ind{index};

    // Duplicated output is not allowed
    if (_graph->getOutputs().contains(ind))
    {
      return false;
    }

    _graph->addOutput(ind);
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}

void ANeuralNetworksModel::allowFloat32toFloat16(bool allow) noexcept
{
  _allowFloat32toFloat16 = allow;
}

bool ANeuralNetworksModel::finish() noexcept
{
  try
  {
    fillOptionalOperand();

    _graph->verify();
    _operand_usages.clear();
    _finished_building = true;
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << '\n';

    return false;
  }

  return true;
}

bool ANeuralNetworksModel::isFinished() noexcept { return _finished_building; }

bool ANeuralNetworksModel::isExistOperand(uint32_t index) noexcept
{
  return _graph->operands().exist(onert::ir::OperandIndex{index});
}

size_t ANeuralNetworksModel::operandSize(uint32_t index) noexcept
{
  try
  {
    return _graph->operands().at(onert::ir::OperandIndex{index}).operandSize();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << '\n';

    return 0;
  }
}

bool ANeuralNetworksModel::isUsageSet(uint32_t index) noexcept
{
  return (_operand_usages[index] != OperandUsage::NOT_DEFINED);
}

bool ANeuralNetworksModel::isOperationOutput(uint32_t index) noexcept
{
  return (_operand_usages[index] == OperandUsage::OPERATION_OUTPUT);
}

void ANeuralNetworksModel::setOptionalOperand(const onert::ir::OperandIndex idx)
{
  _optional_operands.insert(idx);
}

void ANeuralNetworksModel::fillOptionalOperand(void)
{
  _graph->operations().iterate([&](const onert::ir::OperationIndex &, onert::ir::IOperation &node) {
    for (auto &&input : node.getInputs())
    {
      // TODO fill default value for optional operands
      if (_optional_operands.find(input) != _optional_operands.end())
      {
        throw std::runtime_error{"Optional operand is not supported yet"};
      }
    }
  });
}

std::shared_ptr<onert::ir::Model> ANeuralNetworksModel::getModel() const
{
  auto model = std::make_shared<onert::ir::Model>();

  model->push(onert::ir::SubgraphIndex{0}, _graph);
  // TODO Find all child subgraphs and copy them to all_subgs
  // Must find the same subgraph by using to compare pointer of subgraphs and set subgraph's index
  // to operands of control flow operations
  // Must clean all child subgraphs's pointer to prevent memory leak in case of that graph has
  // subgraph itself recursively

  return model;
}
