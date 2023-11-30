/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_COMPUTE_FULLY_CONNECTED_H__
#define __LUCI_COMPUTE_FULLY_CONNECTED_H__

#include "Types.h"

#include <loco/IR/TensorShape.h>

namespace luci
{
namespace compute
{

// TODO extract some common for multiple Ops
class FullyConnected
{
public:
  FullyConnected() = default;

public:
  FullyConnectedParams &params(void) { return _params; }

  bool keep_num_dims(void) const { return _keep_num_dims; }
  void keep_num_dims(bool knd) { _keep_num_dims = knd; }

  void input(const loco::TensorShape &shape, const float *data)
  {
    _input_shape = shape;
    _input_data = data;
  }

  void weights(const loco::TensorShape &shape, const float *data)
  {
    _weights_shape = shape;
    _weights_data = data;
  }

  void bias(const loco::TensorShape &shape, const float *data)
  {
    _bias_shape = shape;
    _bias_data = data;
  }

  void output(float *data) { _output_data = data; }

public:
  bool prepare(void);
  const loco::TensorShape &outputShape(void) const { return _output_shape; }
  void compute(void);

private:
  // param to pass to compute kernel
  FullyConnectedParams _params;
  // new param from tflite version 5
  bool _keep_num_dims = false;
  // shape and data for inputs
  loco::TensorShape _input_shape;
  loco::TensorShape _weights_shape;
  loco::TensorShape _bias_shape;
  const float *_input_data = nullptr;
  const float *_weights_data = nullptr;
  const float *_bias_data = nullptr;

  // compute results
  loco::TensorShape _output_shape;
  float *_output_data = nullptr;
};

} // namespace compute
} // namespace luci

#endif // __LUCI_COMPUTE_FULLY_CONNECTED_H__
