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

#include "SliceLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Slice.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

SliceLayer::SliceLayer() : _input(nullptr), _begin(nullptr), _size(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <typename T>
void SliceLayer::GetBeginAndSizeVectors(int dimensions, const IPortableTensor *begin,
                                        const IPortableTensor *size, std::vector<int> *begins,
                                        std::vector<int> *sizes)
{
  for (int idx = dimensions - 1; idx >= 0; --idx)
  {
    begins->push_back(reinterpret_cast<T *>(begin->buffer())[idx]);
    sizes->push_back(reinterpret_cast<T *>(size->buffer())[idx]);
  }
}

void SliceLayer::sliceFloat32()
{
  const int kMaxDim = nnfw::cker::Shape::kMaxSmallSize;

  std::vector<int> begins;
  std::vector<int> sizes;
  begins.reserve(kMaxDim);
  sizes.reserve(kMaxDim);

  GetBeginAndSizeVectors<int32_t>(_input->num_dimensions(), _begin, _size, &begins, &sizes);

  // begins : 0-based, sizes : 1-based
  for (int i = _input->num_dimensions(); i < kMaxDim; ++i)
  {
    begins.push_back(0);
    sizes.push_back(1);
  }

  nnfw::cker::SliceParams op_params;
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; ++i)
  {
    op_params.begin[i] = begins[3 - i];
    op_params.size[i] = sizes[3 - i];
  }

  nnfw::cker::Slice(op_params, getExtendedTensorShape(_input),
                    reinterpret_cast<const float *>(_input->buffer()),
                    reinterpret_cast<float *>(_output->buffer()));
}

void SliceLayer::sliceQuant8()
{
  // cker quant8 slice is not implemented yet
  throw std::runtime_error{"NYI"};
}

void SliceLayer::configure(const IPortableTensor *input, const IPortableTensor *begin,
                           const IPortableTensor *size, IPortableTensor *output)
{
  _input = input;
  _output = output;
  _begin = begin;
  _size = size;
}

void SliceLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    sliceFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    sliceQuant8();
  }
  else
  {
    throw std::runtime_error{"Slice: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
