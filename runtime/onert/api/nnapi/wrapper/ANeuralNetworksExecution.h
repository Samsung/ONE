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

#ifndef __EXECUTION_H__
#define __EXECUTION_H__

#include <NeuralNetworks.h>

#include <memory>

#include "exec/Execution.h"

struct ANeuralNetworksExecution
{
public:
  ANeuralNetworksExecution(const std::shared_ptr<onert::exec::IExecutors> &executors)
    : _execution{std::make_shared<onert::exec::Execution>(executors)}
  {
    // DO NOTHING
  }

public:
  bool setInput(uint32_t index, const ANeuralNetworksOperandType *type, const void *buffer,
                size_t length) noexcept;
  bool setOptionalInput(uint32_t index, const ANeuralNetworksOperandType *type, const void *buffer,
                        size_t length) noexcept;
  bool setOutput(uint32_t index, const ANeuralNetworksOperandType *type, void *buffer,
                 size_t length) noexcept;
  bool startExecute(void) noexcept;
  bool execute(void) noexcept;

  const onert::ir::OperandIndex getInputOperandIndex(int32_t index) noexcept;
  const onert::ir::OperandIndex getOutputOperandIndex(int32_t index) noexcept;
  bool compareDataType(const ANeuralNetworksOperandType *type,
                       const onert::ir::OperandIndex index) noexcept;
  bool compareShape(const ANeuralNetworksOperandType *type,
                    const onert::ir::OperandIndex index) noexcept;
  bool IsOptionalInput(const onert::ir::OperandIndex index) noexcept;
  bool hasUnspecifiedDims(const onert::ir::OperandIndex index) noexcept;
  size_t getOperandSize(const onert::ir::OperandIndex index) noexcept;
  const std::shared_ptr<onert::exec::Execution> instance(void) noexcept;

  /**
   * @brief       Get output operand's rank
   * @param[in]   index Output index
   * @param[out]  rank  Output operand's rank
   * @return      @c true if success to get rank, otherwise @c false
   */
  bool getOutputOperandRank(uint32_t index, uint32_t *rank) noexcept;
  /**
   * @brief       Get dimensions of the output operand
   * @param[in]   index Output index
   * @param[out]  dimensions  Output operand's dimensions
   * @return      @c true if success to get rank, otherwise @c false
   * @note        This must be called after execution is finished to get resolved output shape
   *              unspecified in model
   */
  bool getOutputOperandDimensions(uint32_t index, uint32_t *dimensions);

private:
  std::shared_ptr<onert::exec::Execution> _execution;
};

#endif
