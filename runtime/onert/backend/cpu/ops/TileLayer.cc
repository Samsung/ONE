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

#include <cker/operation/Tile.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

TileLayer::TileLayer() : _input(nullptr), _multipliers(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void TileLayer::tileFloat32()
{
  TileOneDimension(getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
                   reinterpret_cast<const int *>(_multipliers->buffer()),
                   reinterpret_cast<float *>(_output->buffer()), 0);
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

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
