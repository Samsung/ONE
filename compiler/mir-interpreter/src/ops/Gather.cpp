/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Gather.h"
#include "Common.h"

#include "mir/Tensor.h"

namespace mir_interpreter
{

using namespace mir;

template <typename T, typename IndicesT> struct GatherImpl
{
  static void run(const TensorVariant &datav, const TensorVariant &indicesv,
                  const ops::GatherOp &op, mir::TensorVariant &res);
};

template <typename T, typename IndicesT>
void GatherImpl<T, IndicesT>::run(const TensorVariant &datav, const TensorVariant &indicesv,
                                  const ops::GatherOp &op, TensorVariant &res)
{
  const auto &data_shape = datav.getShape();
  const auto &indices_shape = indicesv.getShape();
  Tensor<T> data(datav);
  Tensor<T> output(res);
  Tensor<IndicesT> indices(indicesv);

  int32_t axis = op.getAxis();
  if (axis < 0)
    axis += data_shape.rank();
  assert(axis >= 0 && axis < data_shape.rank());
  int32_t axis_size = data_shape.dim(axis);
  int32_t num_indices = indices_shape.numElements();

  int32_t outer_size = 1;
  for (int32_t i = 0; i < axis; ++i)
    outer_size *= data_shape.dim(i);

  int32_t inner_size = 1;
  for (int32_t i = axis + 1; i < data_shape.rank(); ++i)
    inner_size *= data_shape.dim(i);

  for (int32_t outer = 0; outer < outer_size; ++outer)
  {
    for (int32_t i = 0; i < num_indices; ++i)
    {
      auto index = indices.atOffset(i);
      assert(index >= 0 && index < axis_size);
      for (int32_t inner = 0; inner < inner_size; inner++)
      {
        output.atOffset((outer * num_indices + i) * inner_size + inner) =
          data.atOffset((outer * axis_size + index) * inner_size + inner);
      }
    }
  }
}

// a hack to reuse dispath function
template <typename T> struct GatherByT
{

  template <typename IndicesT> using GatherWithFixedT = GatherImpl<T, IndicesT>;

  static void run(const TensorVariant &data, const TensorVariant &indices, const ops::GatherOp &op,
                  TensorVariant &res)
  {
    dispatch<GatherWithFixedT>(indices.getElementType(), data, indices, op, res);
  }
};

void Gather(const TensorVariant &data, const TensorVariant &indices, const ops::GatherOp &op,
            TensorVariant &res)
{
  dispatch<GatherByT>(data.getElementType(), data, indices, op, res);
}

} // namespace mir_interpreter
