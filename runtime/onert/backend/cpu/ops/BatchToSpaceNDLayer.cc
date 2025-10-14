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

#include "BatchToSpaceNDLayer.h"

#include <cker/operation/BatchToSpaceND.h>

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::BatchToSpaceND::INPUT)};
  const auto block_size_index{node.getInputs().at(ir::operation::BatchToSpaceND::BLOCK_SIZE)};

  auto output_alloc = _tensor_reg->getPortableTensor(output_index);
  auto input_alloc = _tensor_reg->getPortableTensor(input_index);
  auto block_size_alloc = _tensor_reg->getPortableTensor(block_size_index);

  auto fn = std::make_unique<ops::BatchToSpaceNDLayer>();

  IPortableTensor *crops_alloc = nullptr;
  const auto NNApiInputs = 2;

  if (node.getInputs().size() != NNApiInputs)
  {
    const auto crops_data_index{node.getInputs().at(ir::operation::BatchToSpaceND::CROPS_DATA)};
    crops_alloc = _tensor_reg->getPortableTensor(crops_data_index);
  }

  fn->configure(input_alloc, output_alloc, block_size_alloc, crops_alloc);

  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::BatchToSpaceND &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

BatchToSpaceNDLayer::BatchToSpaceNDLayer()
  : _input(nullptr), _output(nullptr), _block_shape(nullptr), _crops(nullptr)
{
  // DO NOTHING
}

template <typename T> void BatchToSpaceNDLayer::batchToSpaceNDGeneric()
{
  const int32_t NNapiCrops[]{0, 0, 0, 0};
  const int32_t *_crops_buffer;

  if (_crops == nullptr)
  {
    _crops_buffer = NNapiCrops;
  }
  else
  {
    _crops_buffer = getBuffer<int32_t>(_crops);
  }
  nnfw::cker::BatchToSpaceND<T>(getShape(_input), getBuffer<T>(_input),
                                getBuffer<int32_t>(_block_shape), _crops_buffer, getShape(_output),
                                getBuffer<T>(_output));
}

void BatchToSpaceNDLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                    IPortableTensor *block_shape, IPortableTensor *crops)
{
  _output = output;
  _input = input;
  _block_shape = block_shape;
  _crops = crops;
}

void BatchToSpaceNDLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    batchToSpaceNDGeneric<float>();
  }
  else if (_output->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    batchToSpaceNDGeneric<uint8_t>();
  }
  else
  {
    throw std::runtime_error{"NYI"};
  }
}

} // namespace onert::backend::cpu::ops
