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

#include "FusedBatchNormLayer.h"

#include <cker/operation/FusedBatchNorm.h>

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::FusedBatchNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  auto output_tensor = _tensor_reg->getPortableTensor(ofm_index);
  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  const auto epsilon = node.param().epsilon;
  const auto is_training = node.param().is_training;
  const auto &data_format = node.param().data_format;

  auto fn = std::make_unique<ops::FusedBatchNormLayer>();

  fn->configure(input_tensors, epsilon, is_training, data_format, output_tensor);

  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::FusedBatchNorm &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

FusedBatchNormLayer::FusedBatchNormLayer()
  : _inputs(), _output(nullptr), _epsilon(0), _is_training(true),
    _fusedbatchnorm_kernel(new nnfw::cker::FusedBatchNorm())
{
  // DO NOTHING
}

FusedBatchNormLayer::~FusedBatchNormLayer() = default;

void FusedBatchNormLayer::fusedbatchnormFloat32()
{
  uint32_t num_inputs = _inputs.size();
  nnfw::cker::FusedBatchNorm &kernel = *_fusedbatchnorm_kernel;

  kernel.prepare();

  std::vector<nnfw::cker::Shape> inputShapes;
  std::vector<const float *> inputFloatPtrs;

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    inputShapes.emplace_back(getShape(_inputs[i]));
    inputFloatPtrs.emplace_back(getBuffer<float>(_inputs[i]));
  }

  nnfw::cker::FusedBatchNormParams param;

  param.epsilon = _epsilon;
  param.is_training = _is_training;
  param.data_format = _data_format;

  kernel(inputShapes, inputFloatPtrs, getShape(_output), getBuffer<float>(_output), param);
}

void FusedBatchNormLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    fusedbatchnormFloat32();
  }
  else
  {
    throw std::runtime_error{"FusedBatchNorm: unsupported data type"};
  }
}

void FusedBatchNormLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                                    float epsilon, bool is_training, std::string data_format,
                                    IPortableTensor *output)
{
  assert(inputs.size() > 0);
  assert(output != nullptr);

  _inputs = inputs;
  _output = output;
  _epsilon = epsilon;
  _is_training = is_training;
  _data_format = data_format;
}

} // namespace onert::backend::cpu::ops
