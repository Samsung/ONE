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
#include <util/Exceptions.h>

namespace onert::backend::cpu::ops
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
  nnfw::cker::Shape lhs_shape = getShape(_lhs);
  nnfw::cker::Shape rhs_shape = getShape(_rhs);
  nnfw::cker::Shape output_shape = getShape(_output);

  // TODO implement for constant input

  batchmatmul_kernel.prepare(lhs_shape, rhs_shape, _adj_x, _adj_y, _rhs->is_constant());
  batchmatmul_kernel(lhs_shape, getBuffer<float>(_lhs), rhs_shape, getBuffer<float>(_rhs), _adj_x,
                     _adj_y, output_shape, getBuffer<float>(_output));
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
  if (_lhs->data_type() != OperandType::FLOAT32)
    throw UnsupportedDataTypeException{"BatchMatMul", _lhs->data_type()};
  if (_rhs->data_type() != OperandType::FLOAT32)
    throw UnsupportedDataTypeException{"BatchMatMul", _rhs->data_type()};
  batchMatMulFloat32();
}

#undef AVGPOOLING_PARAMETERS

} // namespace onert::backend::cpu::ops
