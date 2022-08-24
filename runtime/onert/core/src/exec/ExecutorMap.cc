
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

#include "exec/ExecutorMap.h"

namespace onert
{
namespace exec
{

uint32_t ExecutorMap::inputSize() const
{
  return _multi_model ? _pkg_inputs.size()
                      : _map.at(ir::SubgraphIndex{0})->graph().getInputs().size();
}

uint32_t ExecutorMap::outputSize() const
{
  return _multi_model ? _pkg_outputs.size()
                      : _map.at(ir::SubgraphIndex{0})->graph().getOutputs().size();
}

const ir::OperandInfo ExecutorMap::inputInfo(const ir::IOIndex &index)
{
  if (_multi_model)
  {
    // Assume that each model may have only one subgraph
    // TODO handle general case
    const auto desc = _pkg_inputs[index.value()];
    const auto model_idx = std::get<0>(desc);
    const auto executor_idx = ir::SubgraphIndex{model_idx.value()};
    const auto input_index = _map.at(executor_idx)->graph().getInputs().at(std::get<2>(desc));
    return _map.at(executor_idx)->graph().operands().at(input_index).info();
  }

  const auto input_index = _map.at(ir::SubgraphIndex{0})->graph().getInputs().at(index);
  return _map.at(ir::SubgraphIndex{0})->graph().operands().at(input_index).info();
}

const ir::OperandInfo ExecutorMap::outputInfo(const ir::IOIndex &index)
{
  if (_multi_model)
  {
    // Assume that each model may have only one subgraph
    // TODO handle general case
    auto desc = _pkg_outputs[index.value()];
    auto model_idx = std::get<0>(desc);
    auto executor_idx = ir::SubgraphIndex{model_idx.value()};
    auto output_index = _map.at(executor_idx)->graph().getOutputs().at(std::get<2>(desc));
    return _map.at(executor_idx)->graph().operands().at(output_index).info();
  }

  auto output_index = _map.at(ir::SubgraphIndex{0})->graph().getOutputs().at(index);
  return _map.at(ir::SubgraphIndex{0})->graph().operands().at(output_index).info();
}

void ExecutorMap::execute(const IODescription &desc)
{
  if (_multi_model)
    throw std::runtime_error{"NYI: Multi model execution is not supported yet"};

  _map.at(ir::SubgraphIndex{0})->execute(desc);
}
} // namespace exec
} // namespace onert
