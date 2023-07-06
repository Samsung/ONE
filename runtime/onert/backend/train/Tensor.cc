/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Tensor.h"

#include <util/Utils.h>

#include <functional>

namespace
{

using namespace onert;

template <typename T, typename L>
void apply(const ir::Shape &shape, const backend::IPortableTensor &src,
           backend::train::TrainableTensor &dst, L &&f)
{
  ShapeLoop(shape, [&](const ir::Coordinates &coords) {
    const T src_val = *reinterpret_cast<const T *>(src.buffer() + src.calcOffset(coords));
    T *dst_data = reinterpret_cast<T *>(dst.buffer() + dst.calcOffset(coords));
    *dst_data = f(src_val, *dst_data);
  });
}

} // namespace

namespace onert
{
namespace backend
{
namespace train
{

// TODO Optimize this method
void TrainableTensor::applyGradient(const IPortableTensor &grad_tensor, double lr)
{
  assert(data_type() == grad_tensor.data_type());

  const auto &shape = get_info().shape();
  const auto &grad_shape = grad_tensor.get_info().shape();

  // TODO Support for different shapes
  if (shape != grad_shape)
  {
    throw std::runtime_error("TrainableTensor: Invalid gradient tensor");
  }

  switch (grad_tensor.data_type())
  {
    case ir::DataType::FLOAT32:
      apply<float>(shape, grad_tensor, *this,
                   [&](float dst, float src) -> float { return dst + src * lr; });
      break;
    default:
      throw std::runtime_error("TrainableTensor: Not supported data type");
  }
}

void TrainableTensor::fillBuffer(const std::shared_ptr<ir::Data> &data)
{
  assert(_buffer);
  assert(total_size() == data->size());
  std::memcpy(_buffer, data->base(), data->size());
}

} // namespace train
} // namespace backend
} // namespace onert
