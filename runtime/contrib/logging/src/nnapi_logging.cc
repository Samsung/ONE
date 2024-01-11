/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <NeuralNetworks.h>
#include <NeuralNetworksEx.h>

#include <stdexcept>
#include <iostream>

#include <string>
#include <map>

#include <cassert>

namespace
{

class OperationCodeResolver
{
public:
  OperationCodeResolver();

public:
  std::string resolve(int code) const;

private:
  void setName(int code, const std::string &name);

private:
  std::map<int, std::string> _table;

public:
  static const OperationCodeResolver &access()
  {
    static const OperationCodeResolver resolver;

    return resolver;
  }
};

OperationCodeResolver::OperationCodeResolver()
{
#define NNAPI_OPERATION(NAME, CODE) setName(CODE, #NAME);
#include "operation.def"
#undef NNAPI_OPERATION
}

void OperationCodeResolver::setName(int code, const std::string &name)
{
  assert(_table.find(code) == _table.end());
  _table[code] = name;
}

std::string OperationCodeResolver::resolve(int code) const
{
  auto it = _table.find(code);

  if (it == _table.end())
  {
    return "unknown(" + std::to_string(code) + ")";
  }

  return it->second;
}

class OperandCodeResolver
{
public:
  OperandCodeResolver();

public:
  std::string resolve(int code) const;

private:
  void setName(int code, const std::string &name);

private:
  std::map<int, std::string> _table;

public:
  static const OperandCodeResolver &access()
  {
    static const OperandCodeResolver resolver;

    return resolver;
  }
};

OperandCodeResolver::OperandCodeResolver()
{
#define NNAPI_OPERAND(NAME, CODE) setName(CODE, #NAME);
#include "operand.def"
#undef NNAPI_OPERAND
}

void OperandCodeResolver::setName(int code, const std::string &name)
{
  assert(_table.find(code) == _table.end());
  _table[code] = name;
}

std::string OperandCodeResolver::resolve(int code) const
{
  auto it = _table.find(code);

  if (it == _table.end())
  {
    return "unknown(" + std::to_string(code) + ")";
  }

  return it->second;
}
} // namespace

//
// Asynchronous Event
//
struct ANeuralNetworksEvent
{
};

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent *event) { return ANEURALNETWORKS_NO_ERROR; }

void ANeuralNetworksEvent_free(ANeuralNetworksEvent *event) { delete event; }

//
// Memory
//
struct ANeuralNetworksMemory
{
  // 1st approach - Store all the data inside ANeuralNetworksMemory object
  // 2nd approach - Store metadata only, and defer data loading as much as possible
};

int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                       ANeuralNetworksMemory **memory)
{
  *memory = new ANeuralNetworksMemory;

  std::cout << __FUNCTION__ << "() --> (memory: " << *memory << ")" << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory *memory)
{
  std::cout << __FUNCTION__ << "(" << memory << ")" << std::endl;
  delete memory;
}

//
// Model
//
struct ANeuralNetworksModel
{
  // ANeuralNetworksModel should be a factory for Graph IR (a.k.a ISA Frontend)
  // TODO Record # of operands
  uint32_t numOperands;

  ANeuralNetworksModel() : numOperands(0)
  {
    // DO NOTHING
  }
};

int ANeuralNetworksModel_create(ANeuralNetworksModel **model)
{
  *model = new ANeuralNetworksModel;

  std::cout << __FUNCTION__ << "(" << model << ") --> (model: " << *model << ")" << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel *model)
{
  std::cout << __FUNCTION__ << "(" << model << ")" << std::endl;

  delete model;
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel *model,
                                    const ANeuralNetworksOperandType *type)
{
  std::cout << __FUNCTION__ << "(model: " << model
            << ", type: " << ::OperandCodeResolver::access().resolve(type->type) << ")"
            << std::endl;

  auto id = model->numOperands;

  std::cout << "  id: " << id << std::endl;
  std::cout << "  rank: " << type->dimensionCount << std::endl;
  for (uint32_t dim = 0; dim < type->dimensionCount; ++dim)
  {
    std::cout << "    dim(" << dim << "): " << type->dimensions[dim] << std::endl;
  }

  model->numOperands += 1;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel *model, int32_t index,
                                         const void *buffer, size_t length)
{
  std::cout << __FUNCTION__ << "(model: " << model << ", index: " << index << ")" << std::endl;

  // TODO Implement this!
  // NOTE buffer becomes invalid after ANeuralNetworksModel_setOperandValue returns

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel *model, int32_t index,
                                                   const ANeuralNetworksMemory *memory,
                                                   size_t offset, size_t length)
{
  std::cout << __FUNCTION__ << "(model: " << model << ", index: " << index << ")" << std::endl;

  // TODO Implement this!

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel *model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t *inputs, uint32_t outputCount,
                                      const uint32_t *outputs)
{
  std::cout << __FUNCTION__ << "(model: " << model
            << ", type: " << ::OperationCodeResolver::access().resolve(type)
            << ", inputCount: " << inputCount << ", outputCount: " << outputCount << ")"
            << std::endl;

  for (uint32_t input = 0; input < inputCount; ++input)
  {
    std::cout << "  input(" << input << "): " << inputs[input] << std::endl;
  }
  for (uint32_t output = 0; output < outputCount; ++output)
  {
    std::cout << "  output(" << output << "): " << outputs[output] << std::endl;
  }

  // TODO Implement this!

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                        ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs)
{
  std::cout << __FUNCTION__ << "(model: " << model << ", type: " << type
            << ", inputCount: " << inputCount << ", outputCount: " << outputCount << ")"
            << std::endl;

  for (uint32_t input = 0; input < inputCount; ++input)
  {
    std::cout << "  input(" << input << "): " << inputs[input] << std::endl;
  }
  for (uint32_t output = 0; output < outputCount; ++output)
  {
    std::cout << "  output(" << output << "): " << outputs[output] << std::endl;
  }

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel *model, uint32_t inputCount,
                                                  const uint32_t *inputs, uint32_t outputCount,
                                                  const uint32_t *outputs)
{
  std::cout << __FUNCTION__ << "(model: " << model << ")" << std::endl;

  for (uint32_t input = 0; input < inputCount; ++input)
  {
    std::cout << "  input(" << input << "): " << inputs[input] << std::endl;
  }
  for (uint32_t output = 0; output < outputCount; ++output)
  {
    std::cout << "  output(" << output << "): " << outputs[output] << std::endl;
  }

  // TODO Implement this!
  // NOTE It seems that this function identifies the input and output of the whole model

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel *model)
{
  std::cout << __FUNCTION__ << "(model: " << model << ")" << std::endl;

  // TODO Implement this!

  return ANEURALNETWORKS_NO_ERROR;
}

//
// Compilation
//
struct ANeuralNetworksCompilation
{
  // ANeuralNetworksCompilation should hold a compiled IR
};

int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  *compilation = new ANeuralNetworksCompilation;

  std::cout << __FUNCTION__ << "(model: " << model << ") --> (compilation: " << *compilation << ")"
            << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  std::cout << __FUNCTION__ << "(compilation: " << compilation << ")" << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

//
// Execution
//
struct ANeuralNetworksExecution
{
  // ANeuralNetworksExecution corresponds to NPU::Interp::Session
};

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation *compilation,
                                    ANeuralNetworksExecution **execution)
{
  *execution = new ANeuralNetworksExecution;

  std::cout << __FUNCTION__ << "(compilation: " << compilation << ") --> (execution: " << *execution
            << ")" << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

// ANeuralNetworksExecution_setInput and ANeuralNetworksExecution_setOutput specify HOST buffer for
// input/output
int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution *execution, int32_t index,
                                      const ANeuralNetworksOperandType *type, const void *buffer,
                                      size_t length)
{
  std::cout << __FUNCTION__ << "(execution: " << execution << ", type: ";

  if (type == nullptr)
    std::cout << "nullptr)" << std::endl;
  else
    std::cout << ::OperandCodeResolver::access().resolve(type->type) << ")" << std::endl;

  // Q: Should we transfer input from HOST to DEVICE here, or in
  // ANeuralNetworksExecution_startCompute?

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution *execution, int32_t index,
                                       const ANeuralNetworksOperandType *type, void *buffer,
                                       size_t length)
{
  std::cout << __FUNCTION__ << "(execution: " << execution << ", type: ";

  if (type == nullptr)
    std::cout << "nullptr)" << std::endl;
  else
    std::cout << ::OperandCodeResolver::access().resolve(type->type) << ")" << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution *execution,
                                          ANeuralNetworksEvent **event)
{
  *event = new ANeuralNetworksEvent;

  std::cout << __FUNCTION__ << "(execution: " << execution << ") --> (event: " << *event << ")"
            << std::endl;

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution *execution)
{
  std::cout << __FUNCTION__ << "(execution: " << execution << ")" << std::endl;

  delete execution;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  std::cout << __FUNCTION__ << "(compilation: " << compilation << ")" << std::endl;
  delete compilation;
}
