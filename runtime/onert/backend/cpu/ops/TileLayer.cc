/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TileLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/Tile.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Tile &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Tile &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Tile::INPUT)};
  const auto multiples_index{node.getInputs().at(ir::operation::Tile::MULTIPLES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto multiples_tensor = _tensor_reg->getPortableTensor(multiples_index);

  auto fn = std::make_unique<ops::TileLayer>();

  fn->configure(input_tensor, multiples_tensor, output_tensor);
  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

TileLayer::TileLayer() : _input(nullptr), _multipliers(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void TileLayer::tileFloat32()
{
  TileOneDimension(getShape(_input), getBuffer<float>(_input), getBuffer<int>(_multipliers),
                   getBuffer<float>(_output), 0);
}

void TileLayer::tileQuant8()
{
  // cker quant8 tile is not implemented yet
  throw std::runtime_error{"NYI"};
}

void TileLayer::configure(const IPortableTensor *input, const IPortableTensor *multipliers,
                          IPortableTensor *output)
{
  _input = input;
  _multipliers = multipliers;
  _output = output;
}

void TileLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    tileFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    tileQuant8();
  }
  else
  {
    throw std::runtime_error{"Tile: unsupported data type"};
  }
}

} // namespace onert::backend::cpu::ops
