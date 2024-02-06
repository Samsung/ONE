/* Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_COMPUTE_STRIDED_SLICE_H__
#define __LUCI_COMPUTE_STRIDED_SLICE_H__

#include "Types.h"

#include <loco/IR/TensorShape.h>

namespace luci
{
namespace compute
{

template <typename T> class StridedSlice
{
public:
  StridedSlice() = default;

public:
  StridedSliceParams &params(void) { return _params; }

  void input(const loco::TensorShape &shape, const T *data)
  {
    _input_shape = shape;
    _input_data = data;
  }

  void begin(const loco::TensorShape &shape, const T *data)
  {
    _begin_shape = shape;
    _begin_data = data;
  }

  void end(const loco::TensorShape &shape, const T *data)
  {
    _end_shape = shape;
    _end_data = data;
  }

  void strides(const loco::TensorShape &shape, const T *data)
  {
    _strides_shape = shape;
    _strides_data = data;
  }

  void output(T *data) { _output_data = data; }

public:
  const loco::TensorShape &output_shape(void) const { return _output_shape; }
  bool prepare(void);
  void compute(void);

private:
  // param to pass to compute kernel
  StridedSliceParams _params = {};
  // shape and data for inputs
  loco::TensorShape _input_shape;
  loco::TensorShape _begin_shape;
  loco::TensorShape _end_shape;
  loco::TensorShape _strides_shape;
  const T *_input_data = nullptr;
  const T *_begin_data = nullptr;
  const T *_end_data = nullptr;
  const T *_strides_data = nullptr;

  // compute results
  loco::TensorShape _output_shape;
  T *_output_data = nullptr;
};

} // namespace compute
} // namespace luci

#endif // __LUCI_COMPUTE_STRIDED_SLICE_H__
