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

#ifndef __OPERATION_FACTORY_H__
#define __OPERATION_FACTORY_H__

#include <unordered_map>

#include "ir/Operands.h"
#include "ir/Operation.h"
#include "NeuralNetworks.h"
#include "NeuralNetworksEx.h"

/**
 * @brief A class to create a onert operation object from NN API input parameters
 */
class OperationFactory
{
public:
  struct Param
  {
    uint32_t input_count;
    const uint32_t *inputs;
    uint32_t output_count;
    const uint32_t *outputs;
  };

public:
  using Generator =
    std::function<onert::ir::Operation *(const OperationFactory::Param &, onert::ir::Operands &)>;

public:
  static OperationFactory &get();

private:
  OperationFactory();

public:
  onert::ir::Operation *create(ANeuralNetworksOperationType, const OperationFactory::Param &param,
                               onert::ir::Operands &operands);
  // TODO add "register" method for separating registration, possibly supporting custom-ops

private:
  std::unordered_map<ANeuralNetworksOperationType, Generator> _map;
};

#endif // __OPERATION_FACTORY_H__
