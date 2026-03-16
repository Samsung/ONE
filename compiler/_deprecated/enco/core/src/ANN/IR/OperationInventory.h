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

#ifndef __ANN_IR_OPERATION_INVENTORY_H__
#define __ANN_IR_OPERATION_INVENTORY_H__

#include "ANN/IR/Operation.h"
#include "ANN/IR/OperandID.h"

#include <initializer_list>

#include <memory>

namespace ann
{

class OperationInventory
{
public:
  void create(Operation::Code code, std::initializer_list<OperandID> inputs,
              std::initializer_list<OperandID> outputs);

public:
  uint32_t count(void) const { return _operations.size(); }

public:
  const Operation *at(uint32_t n) const { return _operations.at(n).get(); }

private:
  std::vector<std::unique_ptr<Operation>> _operations;
};

} // namespace ann

#endif // __ANN_IR_OPERATION_INVENTORY_H__
