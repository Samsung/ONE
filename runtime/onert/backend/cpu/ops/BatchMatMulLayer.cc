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

#include "BatchMatMulLayer.h"

#include <cker/operation/BatchMatMul.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

BatchMatMulLayer::BatchMatMulLayer()
    : _lhs(nullptr), _rhs(nullptr), _output(nullptr), _adj_x(false), _adj_y(false),
      _kernel(new nnfw::cker::BatchMatMul())
{
  // DO NOTHING
}

BatchMatMulLayer::~BatchMatMulLayer() = default;

void BatchMatMulLayer::batchMatMulFloat32()
{
  nnfw::cker::BatchMatMul &batchmatmul_kernel = *_kernel;
  nnfw::cker::Shape lhs_shape = getTensorShape(_lhs);
  nnfw::cker::Shape rhs_shape = getTensorShape(_rhs);
  nnfw::cker::Shape output_shape = getTensorShape(_output);

  // TODO implement for constant input

  batchmatmul_kernel.prepare(lhs_shape, rhs_shape, _adj_x, _adj_y);
  batchmatmul_kernel(lhs_shape, reinterpret_cast<const float *>(_lhs->buffer()), rhs_shape,
                     reinterpret_cast<const float *>(_rhs->buffer()), _adj_x, _adj_y, output_shape,
                     reinterpret_cast<float *>(_output->buffer()));
}

void BatchMatMulLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs, bool adj_x,
                                 bool adj_y, IPortableTensor *output)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _adj_x = adj_x;
  _adj_y = adj_y;
  _output = output;
}

void BatchMatMulLayer::run()
{
  if (_lhs->data_type() == OperandType::FLOAT32)
  {
    batchMatMulFloat32();
  }
  else
  {
    throw std::runtime_error{"BatchMatMul: unsupported data type"};
  }
}

#undef AVGPOOLING_PARAMETERS

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
