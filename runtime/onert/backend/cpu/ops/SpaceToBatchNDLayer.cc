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

#include "SpaceToBatchNDLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SpaceToBatchND.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
SpaceToBatchNDLayer::SpaceToBatchNDLayer()
  : _input(nullptr), _block_shape(nullptr), _padding(nullptr), _output(nullptr)
{
  // DO NOTHING
}

// TO DO : move into shape inferer
void SpaceToBatchNDLayer::checkDimension()
{
  const int kSpatialDimensionNum = 2;
  if (_block_shape->getShape().dim(0) != kSpatialDimensionNum)
  {
    throw std::runtime_error("SpaceToBatchND : block_shape(block_size) tensor's rank is wrong\n");
  }

  // Ensures the input height and width (with padding) is a multiple of block
  // shape height and width.
  for (int dim = 0; dim < kSpatialDimensionNum; ++dim)
  {
    int final_dim_size =
      (_input->getShape().dim(dim + 1) + reinterpret_cast<int32_t *>(_padding->buffer())[dim * 2] +
       reinterpret_cast<int32_t *>(_padding->buffer())[dim * 2 + 1]);

    if (final_dim_size % reinterpret_cast<int32_t *>(_block_shape->buffer())[dim] != 0)
    {
      throw std::runtime_error(
        "SpaceToBatchND : padded input's dimension is not a multiple of block size\n");
    }

    if ((int32_t)_output->getShape().dim(dim + 1) !=
        final_dim_size / reinterpret_cast<int32_t *>(_block_shape->buffer())[dim])
    {
      throw std::runtime_error("SpaceToBatchND : wrong output dimension\n");
    }
  }
}

template <> uint32_t SpaceToBatchNDLayer::getPad<float>() { return 0; }
template <> uint32_t SpaceToBatchNDLayer::getPad<uint8_t>() { return _output->data_zero_point(); }

template <typename T> void SpaceToBatchNDLayer::spaceToBatchND()
{
  checkDimension();

  nnfw::cker::SpaceToBatchParams params;
  params.output_offset = getPad<T>();

  nnfw::cker::SpaceToBatchND(
    params, getTensorShape(_input), reinterpret_cast<const T *>(_input->buffer()),
    getTensorShape(_block_shape), reinterpret_cast<const int32_t *>(_block_shape->buffer()),
    getTensorShape(_padding), reinterpret_cast<const int32_t *>(_padding->buffer()),
    getTensorShape(_output), reinterpret_cast<T *>(_output->buffer()));
}

void SpaceToBatchNDLayer::configure(const IPortableTensor *input,
                                    const IPortableTensor *block_shape,
                                    const IPortableTensor *padding, IPortableTensor *output)
{
  _input = input;
  _block_shape = block_shape;
  _padding = padding;
  _output = output;
}

void SpaceToBatchNDLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    spaceToBatchND<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    spaceToBatchND<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"SpaceToBatchND: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
