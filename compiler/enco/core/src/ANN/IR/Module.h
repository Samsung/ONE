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

#ifndef __ANN_IR_MODULE_H__
#define __ANN_IR_MODULE_H__

#include "ANN/IR/WeightInventory.h"
#include "ANN/IR/OperandInventory.h"
#include "ANN/IR/OperationInventory.h"
#include "ANN/IR/InputList.h"
#include "ANN/IR/OutputList.h"

namespace ann
{

class Module
{
public:
  Module() = default;

public:
  WeightInventory *weight(void) { return &_weight; }
  const WeightInventory *weight(void) const { return &_weight; }

  OperandInventory *operand(void) { return &_operand; }
  const OperandInventory *operand(void) const { return &_operand; }

  OperationInventory *operation(void) { return &_operation; }
  const OperationInventory *operation(void) const { return &_operation; }

  InputList *input(void) { return &_input; }
  const InputList *input(void) const { return &_input; }

  OutputList *output(void) { return &_output; }
  const OutputList *output(void) const { return &_output; }

private:
  WeightInventory _weight;
  OperandInventory _operand;
  OperationInventory _operation;
  InputList _input;
  OutputList _output;
};

} // namespace ann

#endif // __ANN_IR_MODULE_H__
