/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/Tile.h"

#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

Tile::Tile(const Tensor *input, const Tensor *multiples, Tensor *output)
  : Kernel({input, multiples}, {output})
{
}

void Tile::configure()
{
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= 1);
  LUCI_INTERPRETER_CHECK(multiples()->shape().num_dims() == 1);
  LUCI_INTERPRETER_CHECK(multiples()->shape().dim(0) == input()->shape().num_dims());
  LUCI_INTERPRETER_CHECK(multiples()->element_type() == DataType::S32);

  Shape output_shape(input()->shape().num_dims());
  const int32_t *muldata = getTensorData<int32_t>(multiples());
  int32_t num_dim = multiples()->shape().dim(0);
  for (int32_t dim = 0; dim < num_dim; ++dim)
  {
    output_shape.dim(dim) = input()->shape().dim(dim) * muldata[dim];
  }
  output()->resize(output_shape);
}

void Tile::execute() const
{
  switch (output()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

namespace
{

template <typename T, typename M>
void CopyMultipleTimes(const T *in_data, int32_t in_size, M multiplier, T *out_data)
{
  for (M i = 0; i < multiplier; ++i)
  {
    const T *in_end = in_data + in_size;
    T *new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const tflite::RuntimeShape &in_dimensions, const T *in_data,
                                     const M *multiples, T *out_data, int dimension)
{
  if (in_dimensions.DimensionsCount() == 0)
  {
    // If input tensor is a scalar, then just copy it to output (no need to multiply).
    *out_data = *in_data;
    return std::make_pair(0, 0);
  }

  const int dimension_size = in_dimensions.Dims(dimension);
  if (dimension == in_dimensions.DimensionsCount() - 1)
  {
    CopyMultipleTimes(in_data, dimension_size, multiples[dimension], out_data);
    return std::make_pair(dimension_size, dimension_size * static_cast<int>(multiples[dimension]));
  }

  int total_stride_size = 0, total_tiled_stride_size = 0;
  const T *copy_from_data = in_data;
  T *copy_to_data = out_data;
  for (int i = 0; i < dimension_size; ++i)
  {
    int stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
      TileOneDimension(in_dimensions, copy_from_data, multiples, copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size, multiples[dimension] - 1,
                    out_data + total_tiled_stride_size);
  return std::make_pair(total_stride_size,
                        static_cast<int>(total_tiled_stride_size * multiples[dimension]));
}

} // namespace

void Tile::evalFloat() const
{
  TileOneDimension(getTensorShape(input()), getTensorData<float>(input()),
                   getTensorData<int32_t>(multiples()), getTensorData<float>(output()), 0);
}

} // namespace kernels
} // namespace luci_interpreter
