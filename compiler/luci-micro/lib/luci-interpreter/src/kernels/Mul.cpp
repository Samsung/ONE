/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Mul.h"

#include "kernels/Utils.h"

// #include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
#include <tensorflow\lite\kernels\internal\reference\process_broadcast_shapes.h>
#include <tensorflow\lite\kernels\internal\reference\mul.h>
#include <stdexcept>

namespace luci_interpreter
{
  namespace kernels
  {

    Mul::Mul(const Tensor *input1, const Tensor *input2, Tensor *output, const MulParams &params)
        : KernelWithParams<MulParams>(params), _input1(input1), _input2(input2), _output(output)
    {
    }

    void Mul::configure()
    {
      assert(_input1->element_type() == _input2->element_type());
      _output->resize(calculateShapeForBroadcast(_input1->shape(), _input2->shape()));
    }

    void Mul::execute() const
    {
      switch (_input1->element_type())
      {
      case DataType::FLOAT32:
        evalFloat();
        break;
      default:
        throw std::runtime_error("Unsupported type.");
      }
    }

    void Mul::evalFloat() const
    {
      float activation_min{};
      float activation_max{};
      calculateActivationRange(_params.activation, &activation_min, &activation_max);

      tflite::ArithmeticParams params{};
      params.float_activation_min = activation_min;
      params.float_activation_max = activation_max;

      const bool need_broadcast = tflite::reference_ops::ProcessBroadcastShapes(
          getTensorShape(_input1), getTensorShape(_input2), &params);

      if (need_broadcast)
      {
        tflite::reference_ops::BroadcastMul4DSlow(
            params, getTensorShape(_input1), getTensorData<float>(_input1), getTensorShape(_input2),
            getTensorData<float>(_input2), getTensorShape(_output), getTensorData<float>(_output));
      }
      else
      {
        tflite::reference_ops::Mul(params, getTensorShape(_input1), getTensorData<float>(_input1),
                                   getTensorShape(_input2), getTensorData<float>(_input2),
                                   getTensorShape(_output), getTensorData<float>(_output));
      }
    }

  } // namespace kernels
} // namespace luci_interpreter
