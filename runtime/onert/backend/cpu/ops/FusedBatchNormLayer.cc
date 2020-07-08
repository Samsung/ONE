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

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
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
    inputShapes.emplace_back(getTensorShape(_inputs[i]));
    inputFloatPtrs.emplace_back(reinterpret_cast<const float *>(_inputs[i]->buffer()));
  }

  nnfw::cker::FusedBatchNormParams param;

  param.epsilon = _epsilon;
  param.is_training = _is_training;
  param.data_format = _data_format;

  kernel(inputShapes, inputFloatPtrs, getTensorShape(_output),
         reinterpret_cast<float *>(_output->buffer()), param);
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

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
