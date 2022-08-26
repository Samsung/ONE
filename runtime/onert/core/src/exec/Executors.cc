/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/Executors.h"

namespace onert
{
namespace exec
{

uint32_t Executors::inputSize() const
{
  return _executors.at(ir::SubgraphIndex{0})->graph().getInputs().size();
}

uint32_t Executors::outputSize() const
{
  return _executors.at(ir::SubgraphIndex{0})->graph().getOutputs().size();
}

const ir::OperandInfo Executors::inputInfo(const ir::IOIndex &index)
{
  const auto input_index = _executors.at(ir::SubgraphIndex{0})->graph().getInputs().at(index);
  return _executors.at(ir::SubgraphIndex{0})->graph().operands().at(input_index).info();
}

const ir::OperandInfo Executors::outputInfo(const ir::IOIndex &index)
{
  auto output_index = _executors.at(ir::SubgraphIndex{0})->graph().getOutputs().at(index);
  return _executors.at(ir::SubgraphIndex{0})->graph().operands().at(output_index).info();
}

void Executors::execute(const IODescription &desc)
{
  _executors.at(ir::SubgraphIndex{0})->execute(desc);
}

} // namespace exec
} // namespace onert
