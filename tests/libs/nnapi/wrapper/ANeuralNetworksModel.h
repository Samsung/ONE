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

#ifndef __MODEL_H__
#define __MODEL_H__

#include <unordered_set>
#include <NeuralNetworks.h>
#include <NeuralNetworksEx.h>

#include "ir/Graph.h"
#include "ir/Model.h"

struct ANeuralNetworksModel
{
public:
  enum class OperandUsage
  {
    NOT_DEFINED = 0,
    MODEL_INPUT,
    CONSTANT,
    OPERATION_OUTPUT,
  };

public:
  ANeuralNetworksModel() noexcept;

public:
  bool addOperand(const ANeuralNetworksOperandType *type) noexcept;
  bool setOperandValue(uint32_t index, const void *buffer, size_t length, bool optional = false,
                       bool copy = false) noexcept;
  bool addOperation(ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t *inputs,
                    uint32_t outputCount, const uint32_t *outputs) noexcept;
  bool addOperationEx(ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                      const uint32_t *inputs, uint32_t outputCount,
                      const uint32_t *outputs) noexcept;
  bool addModelInput(uint32_t index) noexcept;
  bool addModelOutput(uint32_t index) noexcept;
  void allowFloat32toFloat16(bool allow) noexcept;
  bool allowedToFp16() const noexcept { return _allowFloat32toFloat16; }
  bool finish() noexcept;

  onert::ir::Graph &deref(void) { return *_graph; }
  bool isFinished() noexcept;
  bool isExistOperand(uint32_t index) noexcept;
  size_t operandSize(uint32_t index) noexcept;
  bool isUsageSet(uint32_t index) noexcept;
  bool isOperationOutput(uint32_t index) noexcept;
  std::shared_ptr<onert::ir::Model> getModel() const;

private:
  void setOptionalOperand(const onert::ir::OperandIndex idx);
  void fillOptionalOperand(void);

private:
  std::shared_ptr<onert::ir::Graph> _graph;
  bool _finished_building;
  std::unordered_set<onert::ir::OperandIndex> _optional_operands;
  std::vector<OperandUsage> _operand_usages;
  bool _allowFloat32toFloat16;
};

#endif // __MODEL_H__
