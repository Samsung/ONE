/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_CORE_RUNTIME_CONTEXT_H
#define ONERT_MICRO_CORE_RUNTIME_CONTEXT_H

#include "OMStatus.h"

#include "reader/OMCircleReader.h"

#include <cstdint>

namespace onert_micro
{
namespace core
{

class OMRuntimeContext
{
private:
  reader::OMCircleReader _reader;

public:
  OMRuntimeContext() = default;
  OMRuntimeContext(const OMRuntimeContext &) = delete;
  OMRuntimeContext &operator=(const OMRuntimeContext &) = delete;
  OMRuntimeContext &&operator=(const OMRuntimeContext &&) = delete;
  OMRuntimeContext(OMRuntimeContext &&) = default;
  ~OMRuntimeContext() = default;

  OMStatus setModel(const char *model_ptr, uint32_t graph_index)
  {
    OMStatus status;
    status = _reader.parse(model_ptr);
    if (status != Ok)
      return status;
    status = _reader.select_subgraph(graph_index);
    if (status != Ok)
      return status;
    return Ok;
  }

  const bool isConstTensor(uint32_t tensor_index) { return _reader.isConstTensor(tensor_index); }

  const reader::CircleValues *getCircleOutputs() { return _reader.outputs(); }
  const reader::CircleValues *getCircleInputs() { return _reader.inputs(); }
  const reader::CircleTensors *getCircleTensors() { return _reader.tensors(); }

  const reader::CircleOperators *getCircleOperators();

  const reader::CircleOperatorCodes *getCircleOpcodes() { return _reader.opcodes(); }

  const circle::Operator *getCircleOperatorAt(uint16_t index);
  const circle::Tensor *getTensorByIndex(int32_t tensor_index);

  uint32_t getGraphInputTensorIndex(uint32_t index);
  uint32_t getGraphOutputTensorIndex(uint32_t index);

  OMStatus getConstDataByTensorIndex(uint8_t **data, uint16_t tensor_index);
};

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_RUNTIME_CONTEXT_H
