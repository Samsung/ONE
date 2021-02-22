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

#include "OperationInventory.h"

#include <memory>

using std::make_unique;

namespace ann
{

void OperationInventory::create(Operation::Code code, std::initializer_list<OperandID> inputs,
                                std::initializer_list<OperandID> outputs)
{
  _operations.emplace_back(make_unique<Operation>(code, inputs, outputs));
}

} // namespace ann
