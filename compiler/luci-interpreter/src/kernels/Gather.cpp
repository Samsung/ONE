/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Gather.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/common.h>

#include <stdexcept>
#include <cassert>

// NOTE code is copied here cause "ruy::profiler::ScopeLabel label("Gather");" exist
// but "#include "ruy/profiler/instrumentation.h"" line is absent and compile fails.
// TensorFlow doesn't seem to use tflite::reference_ops::Gather() alone.
// TODO include header file and remove reference_ops block after this is fixed

namespace tflite
{
namespace reference_ops
{

template <typename T, typename CoordsT = int32>
inline void Gather(const tflite::GatherParams &op_params, const RuntimeShape &input_shape,
                   const T *input_data, const RuntimeShape &coords_shape,
                   const CoordsT *coords_data, const RuntimeShape &, T *output_data)
{
  int axis = op_params.axis;
  if (axis < 0)
  {
    axis += input_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_shape.DimensionsCount());

  int batch_dims = op_params.batch_dims;
  if (batch_dims < 0)
  {
    batch_dims += coords_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(batch_dims, 0);
  TFLITE_DCHECK_LT(batch_dims, input_shape.DimensionsCount());
  TFLITE_DCHECK_LE(batch_dims, coords_shape.DimensionsCount());
  TFLITE_DCHECK_GE(axis, batch_dims);
  for (int i = 0; i < batch_dims; ++i)
  {
    TFLITE_DCHECK_EQ(input_shape.Dims(i), coords_shape.Dims(i));
  }

  const int axis_size = input_shape.Dims(axis);

  int batch_size = 1;
  for (int i = 0; i < batch_dims; ++i)
  {
    batch_size *= input_shape.Dims(i);
  }

  int outer_size = 1;
  for (int i = batch_dims; i < axis; ++i)
  {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i)
  {
    inner_size *= input_shape.Dims(i);
  }

  int coord_size = 1;
  for (int i = batch_dims; i < coords_shape.DimensionsCount(); ++i)
  {
    coord_size *= coords_shape.Dims(i);
  }

  for (int batch = 0; batch < batch_size; ++batch)
  {
    for (int outer = 0; outer < outer_size; ++outer)
    {
      for (int i = 0; i < coord_size; ++i)
      {
        TFLITE_DCHECK_GE(coords_data[i], 0);
        TFLITE_DCHECK_LT(coords_data[i], axis_size);
        // TODO(rsun): replace memcpy with a for loop
        std::memcpy(output_data + (((batch * outer_size) + outer) * coord_size + i) * inner_size,
                    input_data + (((batch * outer_size) + outer) * axis_size +
                                  coords_data[batch * coord_size + i]) *
                                   inner_size,
                    sizeof(T) * inner_size);
      }
    }
  }
}

} // namespace reference_ops
} // namespace tflite

namespace luci_interpreter
{

namespace kernels
{

Gather::Gather(const Tensor *params, const Tensor *indices, Tensor *output,
               const GatherParams &gparams)
  : KernelWithParams<GatherParams>({params, indices}, {output}, gparams)
{
}

void Gather::configure()
{
  if (params()->element_type() == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }

  LUCI_INTERPRETER_CHECK(indices()->element_type() == DataType::S32 ||
                         indices()->element_type() == DataType::S64);

  // refer tensorflow/lite/kernels/gather.cc

  const Shape &params_shape = params()->shape();
  const Shape &indices_shape = indices()->shape();

  int axis = _params.axis;
  if (axis < 0)
  {
    axis += params_shape.num_dims();
  }
  LUCI_INTERPRETER_CHECK(0 <= axis && axis < params_shape.num_dims());

  int batch_dims = _params.batch_dims;
  // batch_dims should be in range: [-rank(indices), rank(indices)].
  // Negative batch_dims is added with rank of positions.
  if (batch_dims < 0)
  {
    batch_dims += indices_shape.num_dims();
  }
  LUCI_INTERPRETER_CHECK(batch_dims <= axis);
  LUCI_INTERPRETER_CHECK(0 <= batch_dims && batch_dims < params_shape.num_dims());
  LUCI_INTERPRETER_CHECK(batch_dims <= indices_shape.num_dims());
  for (int i = 0; i < batch_dims; ++i)
  {
    LUCI_INTERPRETER_CHECK(params_shape.dim(i) == indices_shape.dim(i));
  }

  const int num_dimensions = params_shape.num_dims() + indices_shape.num_dims() - 1 - batch_dims;

  Shape output_shape(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i)
  {
    output_shape.dim(output_index++) = params_shape.dim(i);
  }
  for (int i = batch_dims; i < indices_shape.num_dims(); ++i)
  {
    output_shape.dim(output_index++) = indices_shape.dim(i);
  }
  for (int i = axis + 1; i < params_shape.num_dims(); ++i)
  {
    output_shape.dim(output_index++) = params_shape.dim(i);
  }
  output()->resize(output_shape);
}

void Gather::execute() const
{
  switch (params()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Gather::evalFloat() const
{
  assert(indices()->element_type() == DataType::S32 || indices()->element_type() == DataType::S64);

  const auto params_data = getTensorData<float>(params());
  auto output_data = getTensorData<float>(output());

  tflite::GatherParams tparams;
  tparams.axis = _params.axis;
  tparams.batch_dims = _params.batch_dims;

  if (indices()->element_type() == DataType::S32)
  {
    const auto indices_data = getTensorData<int32_t>(indices());

    tflite::reference_ops::Gather<float, int32_t>(tparams, getTensorShape(params()), params_data,
                                                  getTensorShape(indices()), indices_data,
                                                  getTensorShape(output()), output_data);
  }
  else
  {
    const auto indices_data = getTensorData<int64_t>(indices());

    tflite::reference_ops::Gather<float, int64_t>(tparams, getTensorShape(params()), params_data,
                                                  getTensorShape(indices()), indices_data,
                                                  getTensorShape(output()), output_data);
  }
}

} // namespace kernels
} // namespace luci_interpreter
